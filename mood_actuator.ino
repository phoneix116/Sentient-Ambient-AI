#include <Adafruit_NeoPixel.h>

// Cable-only mode: no Wi-Fi. Control via USB Serial.

// --- LED Strip Configuration (WS2812B + Adafruit NeoPixel) ---
// If colors look wrong, change the order from NEO_GRB to NEO_RGB or NEO_BRG.
#define LED_PIN       15           // Data pin connected to DIN of the LED strip
#define NUM_LEDS      24           // Set to your strip length
#define BRIGHTNESS    40           // Default brightness (0-255); change at runtime with BRIGHT:<0-255>
#define NEO_ORDER     NEO_GRB      // NEO_GRB is typical for WS2812B

Adafruit_NeoPixel strip(NUM_LEDS, LED_PIN, NEO_ORDER + NEO_KHZ800);

// Track current color for smooth fades
struct RGB { uint8_t r; uint8_t g; uint8_t b; };
static RGB currentColor = {255, 255, 255}; // start white to match previous default

// Helper to print macro names as strings
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// --- Serial Protocol ---
// Host sends lines like: EMOTION:happy\n
// Valid emotions: happy, sad, angry, neutral, fear, disgust, surprise

// --- State Variables for Transitions ---
String currentMood = "neutral";
// Music is played on the host laptop. Only LEDs are controlled here.

// --- Helper Functions (NeoPixel) ---
static inline uint32_t Color(uint8_t r, uint8_t g, uint8_t b) { return strip.Color(r, g, b); }

void fillSolid(uint8_t r, uint8_t g, uint8_t b) {
  uint32_t c = Color(r, g, b);
  for (int i = 0; i < NUM_LEDS; i++) strip.setPixelColor(i, c);
}

// Smooth transition from currentColor to target over steps
void fadeToColor(uint8_t tr, uint8_t tg, uint8_t tb, uint16_t steps = 120, uint8_t msDelay = 4) {
  RGB start = currentColor;
  for (uint16_t i = 0; i <= steps; i++) {
    uint8_t r = (uint8_t)(start.r + (int32_t)(tr - start.r) * i / steps);
    uint8_t g = (uint8_t)(start.g + (int32_t)(tg - start.g) * i / steps);
    uint8_t b = (uint8_t)(start.b + (int32_t)(tb - start.b) * i / steps);
    fillSolid(r, g, b);
    strip.show();
    delay(msDelay);
  }
  currentColor = {tr, tg, tb};
}

// --- Music handling removed (played on laptop) ---

void bootTest() {
  // Quick power-on test: cycle primary colors to verify wiring
  fillSolid(255, 0, 0); strip.show(); delay(400);
  fillSolid(0, 255, 0); strip.show(); delay(400);
  fillSolid(0, 0, 255); strip.show(); delay(400);
  // End in black to avoid high inrush on white if using USB power only
  fillSolid(0, 0, 0); strip.show();
}

void setup() {
  Serial.begin(115200);
  // Initialize LED Strip (Adafruit NeoPixel)
  strip.begin();
  strip.setBrightness(BRIGHTNESS);
  fillSolid(currentColor.r, currentColor.g, currentColor.b);
  strip.show();
  Serial.println("READY");
  Serial.print("LIB=Adafruit_NeoPixel");
  Serial.print(", ORDER="); Serial.print(STR(NEO_ORDER));
  Serial.print(", PIN="); Serial.print(LED_PIN);
  Serial.print(", NUM_LEDS="); Serial.println(NUM_LEDS);
  Serial.print("BRIGHTNESS="); Serial.println(BRIGHTNESS);
  Serial.println("Send EMOTION:<happy|sad|angry|neutral|fear> or TEST or BRIGHT:<0-255>");

  bootTest();
}

void handleSerialCommand(const String &line) {
  String s = line;
  s.trim();
  if (!s.startsWith("EMOTION:")) return;
  String mood = s.substring(8);
  mood.trim();
  if (mood.length() == 0 || mood == currentMood) return;

  Serial.print("Changing mood to: ");
  Serial.println(mood);
  currentMood = mood;

  if (mood == "happy") {
    fadeToColor(255, 255, 0);
  } else if (mood == "sad") {
    fadeToColor(0, 0, 255);
  } else if (mood == "angry") {
    fadeToColor(255, 0, 0);
  } else if (mood == "neutral") {
    fadeToColor(255, 255, 255);
  } else if (mood == "fear") { // 'fear' is the proxy for 'stressed'
    fadeToColor(0, 255, 255);
  } else if (mood == "disgust") {
    fadeToColor(0, 255, 128);
  } else if (mood == "surprise") {
    fadeToColor(255, 0, 255);
  } else {
    Serial.println("Unknown emotion");
  }
}

void loop() {
  static String buf = "";
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (buf.length() > 0) {
        // Simple command router: EMOTION:..., TEST, CHASE, BRIGHT:<0-255>
        String cmd = buf;
        cmd.trim();
        if (cmd.equalsIgnoreCase("TEST")) {
          bootTest();
        } else if (cmd.equalsIgnoreCase("CHASE")) {
          // Low-power chase to count LEDs and verify data path
          uint8_t prev = BRIGHTNESS; // not readable; we dynamically lower then restore
          strip.setBrightness((prev < 60) ? prev : 60);
          for (int i = 0; i < NUM_LEDS; i++) {
            fillSolid(0, 0, 0);
            strip.setPixelColor(i, Color(255, 255, 0));
            strip.show();
            delay(40);
          }
          fillSolid(0, 0, 0); strip.show();
          strip.setBrightness(BRIGHTNESS);
          Serial.println("CHASE done");
        } else if (cmd.startsWith("BRIGHT:")) {
          int val = cmd.substring(7).toInt();
          val = constrain(val, 0, 255);
          strip.setBrightness(val);
          strip.show();
          Serial.print("Brightness set to "); Serial.println(val);
        } else {
          handleSerialCommand(buf);
        }
        buf = "";
      }
    } else {
      if (buf.length() < 200) buf += c; // simple guard
    }
  }
}