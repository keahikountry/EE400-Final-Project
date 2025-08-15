/* Edge Impulse ingestion SDK — camera HEX dump on threshold (one-line HEX)
 * Modeled after your example.
 */

/* Includes ---------------------------------------------------------------- */
#include <FinalProject3_inferencing.h>     // <-- your camera EI library
#include <Arduino_OV767X.h>                // https://www.arduino.cc/reference/en/libraries/arduino_ov767x/
#include <stdint.h>
#include <stdlib.h>

/* ---------- Config ---------- */
#define POS_LABEL        "person"     // <-- change to your positive class label
#define PERSON_THRESH    0.50f        // 50% threshold
#define LOOP_PERIOD_MS   2000         // capture+infer every 2 seconds
/* ---------------------------- */

/* Constant variables ------------------------------------------------------- */
#define EI_CAMERA_RAW_FRAME_BUFFER_COLS 160
#define EI_CAMERA_RAW_FRAME_BUFFER_ROWS 120
#define DWORD_ALIGN_PTR(a)   ((a & 0x3) ?(((uintptr_t)a + 0x4) & ~(uintptr_t)0x3) : a)

/* Edge Impulse ------------------------------------------------------------- */
class OV7675 : public OV767X {
  public:
    int begin(int resolution, int format, int fps);
    void readFrame(void* buffer);
  private:
    int vsyncPin, hrefPin, pclkPin, xclkPin;
    volatile uint32_t* vsyncPort; uint32_t vsyncMask;
    volatile uint32_t* hrefPort;  uint32_t hrefMask;
    volatile uint32_t* pclkPort;  uint32_t pclkMask;
    uint16_t width, height;
    uint8_t bytes_per_pixel;
    uint16_t bytes_per_row;
    uint8_t buf_rows;
    uint16_t buf_size;
    uint8_t resize_height;
    uint8_t *raw_buf; void *buf_mem; uint8_t *intrp_buf; uint8_t *buf_limit;
    void readBuf();
    int allocate_scratch_buffs();
    int deallocate_scratch_buffs();
};

typedef struct { size_t width; size_t height; } ei_device_resize_resolutions_t;

/* Serial helpers (as in example) ------------------------------------------ */
int  ei_get_serial_available(void) { return Serial.available(); }
char ei_get_serial_byte(void)      { return Serial.read(); }

/* Private variables -------------------------------------------------------- */
static OV7675 Cam;
static bool is_initialised = false;

/* Points to the resized/cropped RGB565 output used for EI input */
static uint8_t *ei_camera_capture_out = NULL;
uint32_t resize_col_sz, resize_row_sz;
bool do_resize = false, do_crop = false;

static bool debug_nn = false;

/* Prototypes --------------------------------------------------------------- */
bool ei_camera_init(void);
void ei_camera_deinit(void);
bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf);
int  calculate_resize_dimensions(uint32_t out_width, uint32_t out_height,
                                 uint32_t *resize_col_sz, uint32_t *resize_row_sz, bool *do_resize);
void resizeImage(int srcWidth, int srcHeight, uint8_t *srcImage,
                 int dstWidth, int dstHeight, uint8_t *dstImage, int iBpp);
void cropImage(int srcWidth, int srcHeight, uint8_t *srcImage,
               int startX, int startY, int dstWidth, int dstHeight,
               uint8_t *dstImage, int iBpp);
int  ei_camera_cutout_get_data(size_t offset, size_t length, float *out_ptr);

/* --- ONE-LINE HEX DUMP (no spaces/newlines in payload) ------------------- */
static void dump_hex_one_line(const uint8_t* data, size_t n) {
  static const char HEX[] = "0123456789ABCDEF";
  // print payload only (no newline) in manageable chunks:
  const size_t CHUNK = 256;
  char buf[CHUNK * 2 + 1];
  size_t i = 0;
  while (i < n) {
    size_t m = (n - i > CHUNK) ? CHUNK : (n - i);
    for (size_t k = 0; k < m; ++k) {
      uint8_t b = data[i + k];
      buf[2*k]   = HEX[(b >> 4) & 0xF];
      buf[2*k+1] = HEX[b & 0xF];
    }
    buf[2*m] = 0;
    Serial.print(buf);              // no spaces, no newline
    i += m;
  }
}

/* Arduino setup ------------------------------------------------------------ */
void setup() {
  Serial.begin(115200);
  while (!Serial);
  Serial.println("EI Camera inferencing + one-line HEX dump on threshold");

  ei_printf("Inferencing settings:\n");
  ei_printf("\tImage resolution: %dx%d\n", EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT);
  ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
  ei_printf("\tNo. of classes: %d\n",
            sizeof(ei_classifier_inferencing_categories) /
            sizeof(ei_classifier_inferencing_categories[0]));
}

/* Main loop --------------------------------------------------------------- */
void loop() {
  static uint32_t last = 0;
  uint32_t now = millis();
  if (now - last < LOOP_PERIOD_MS) { delay(5); return; }
  last = now;

  ei_printf("\nTaking photo...\n");

  if (!ei_camera_init()) {
    ei_printf("ERR: Failed to initialize image sensor\r\n");
    return;
  }

  // choose resize dimensions
  uint32_t resize_col_sz_local, resize_row_sz_local;
  bool do_resize_local = false;
  int rdim = calculate_resize_dimensions(
      EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT,
      &resize_col_sz_local, &resize_row_sz_local, &do_resize_local);
  if (rdim) {
    ei_printf("ERR: Failed to calculate resize dimensions (%d)\r\n", rdim);
    ei_camera_deinit(); return;
  }

  // allocate output buffer (RGB565) for EI input size
  void *snapshot_mem = ei_malloc(resize_col_sz_local * resize_row_sz_local * 2);
  if (!snapshot_mem) {
    ei_printf("ERR: alloc snapshot_mem\r\n");
    ei_camera_deinit(); return;
  }
  uint8_t *snapshot_buf = (uint8_t*)DWORD_ALIGN_PTR((uintptr_t)snapshot_mem);

  // capture into snapshot_buf
  if (!ei_camera_capture(EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT, snapshot_buf)) {
    ei_printf("ERR: capture\r\n");
    ei_free(snapshot_mem); ei_camera_deinit(); return;
  }

  // Build EI signal from cutout (as in example)
  ei::signal_t signal;
  signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
  signal.get_data = &ei_camera_cutout_get_data;

  // Run inference
  ei_impulse_result_t result = { 0 };
  EI_IMPULSE_ERROR ei_error = run_classifier(&signal, &result, debug_nn);
  if (ei_error != EI_IMPULSE_OK) {
    ei_printf("Failed to run impulse (%d)\n", ei_error);
    ei_free(snapshot_mem); ei_camera_deinit(); return;
  }

  // Print timing and predictions
  ei_printf("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.): \n",
            result.timing.dsp, result.timing.classification, result.timing.anomaly);

  float human_score = 0.0f;

#if EI_CLASSIFIER_OBJECT_DETECTION == 1
  ei_printf("Object detection bounding boxes:\r\n");
  for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
    ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
    if (bb.value == 0) continue;
    ei_printf("  %s (%f) [ x:%u y:%u w:%u h:%u ]\r\n",
              bb.label, bb.value, bb.x, bb.y, bb.width, bb.height);
    if (strcmp(bb.label, POS_LABEL) == 0 && bb.value > human_score) {
      human_score = bb.value;
    }
  }
#else
  ei_printf("Predictions:\r\n");
  for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
    float v = result.classification[i].value;
    const char* lab = ei_classifier_inferencing_categories[i];
    ei_printf("  %s: %.5f\r\n", lab, v);
    if (strcmp(lab, POS_LABEL) == 0) human_score = v;
  }
#endif

#if EI_CLASSIFIER_HAS_ANOMALY
  ei_printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif

  // Always print the score so you can see it each cycle
  ei_printf("score(%s)=%.3f\r\n", POS_LABEL, human_score);

  // If ≥ threshold, dump entire RGB565 buffer as ONE single hex line
  if (human_score >= PERSON_THRESH) {
    const uint32_t w = EI_CLASSIFIER_INPUT_WIDTH;
    const uint32_t h = EI_CLASSIFIER_INPUT_HEIGHT;
    const uint32_t nbytes = w * h * 2; // RGB565

    // Header line
    ei_printf("BEGIN_FRAME_HEX RGB565 %u %u %u\r\n", w, h, nbytes);

    // One-line HEX payload (no spaces/newlines)
    dump_hex_one_line(snapshot_buf, nbytes);
    Serial.println();                 // newline to end the one-line hex

    // Footer line
    ei_printf("END_FRAME_HEX\r\n");
  }

  // allow stop via serial 'b'
  while (ei_get_serial_available() > 0) {
    if (ei_get_serial_byte() == 'b') {
      ei_printf("Inferencing stopped by user\r\n");
      break;
    }
  }

  ei_free(snapshot_mem);
  ei_camera_deinit();
}

/* ---------- helpers from your example ---------- */
int calculate_resize_dimensions(uint32_t out_width, uint32_t out_height,
                                uint32_t *resize_col_sz, uint32_t *resize_row_sz, bool *do_resize)
{
  size_t list_size = 2;
  const ei_device_resize_resolutions_t list[list_size] = { {42,32}, {128,96} };
  *resize_col_sz = EI_CAMERA_RAW_FRAME_BUFFER_COLS;
  *resize_row_sz = EI_CAMERA_RAW_FRAME_BUFFER_ROWS;
  *do_resize = false;
  for (size_t ix = 0; ix < list_size; ix++) {
    if ((out_width <= list[ix].width) && (out_height <= list[ix].height)) {
      *resize_col_sz = list[ix].width;
      *resize_row_sz = list[ix].height;
      *do_resize = true;
      break;
    }
  }
  return 0;
}

bool ei_camera_init(void) {
  if (is_initialised) return true;
  if (!Cam.begin(QQVGA, RGB565, 1)) {
    ei_printf("ERR: Failed to initialize camera\r\n");
    return false;
  }
  is_initialised = true;
  return true;
}

void ei_camera_deinit(void) {
  if (is_initialised) { Cam.end(); is_initialised = false; }
}

bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf)
{
  if (!is_initialised) { ei_printf("ERR: Camera is not initialized\r\n"); return false; }
  if (!out_buf)        { ei_printf("ERR: invalid parameters\r\n");        return false; }

  int res = calculate_resize_dimensions(img_width, img_height, &resize_col_sz, &resize_row_sz, &do_resize);
  if (res) { ei_printf("ERR: Failed to calculate resize dimensions (%d)\r\n", res); return false; }

  do_crop = (img_width != resize_col_sz) || (img_height != resize_row_sz);

  Cam.readFrame(out_buf); // captures & resizes

  if (do_crop) {
    uint32_t crop_col_start = (resize_col_sz - img_width) / 2;
    uint32_t crop_row_start = (resize_row_sz - img_height) / 2;
    cropImage(resize_col_sz, resize_row_sz,
              out_buf,
              crop_col_start, crop_row_start,
              img_width, img_height,
              out_buf,
              16);
  }
  ei_camera_capture_out = out_buf;
  return true;
}

int ei_camera_cutout_get_data(size_t offset, size_t length, float *out_ptr) {
  size_t pixel_ix = offset * 2;
  size_t bytes_left = length, out_ptr_ix = 0;
  while (bytes_left != 0) {
    uint16_t pixel = (ei_camera_capture_out[pixel_ix] << 8) | ei_camera_capture_out[pixel_ix+1];
    uint8_t r = ((pixel >> 11) & 0x1f) << 3;
    uint8_t g = ((pixel >> 5)  & 0x3f) << 2;
    uint8_t b = ( pixel        & 0x1f) << 3;
    float pixel_f = (r << 16) + (g << 8) + b;
    out_ptr[out_ptr_ix++] = pixel_f;
    pixel_ix += 2; bytes_left--;
  }
  return 0;
}

/* === Resize / Crop implementations (from your example) === */
#ifdef __ARM_FEATURE_SIMD32
#include <device.h>
#endif
#define FRAC_BITS 14
#define FRAC_VAL (1<<FRAC_BITS)
#define FRAC_MASK (FRAC_VAL - 1)

void resizeImage(int srcWidth, int srcHeight, uint8_t *srcImage,
                 int dstWidth, int dstHeight, uint8_t *dstImage, int iBpp)
{
  uint32_t src_x_accum, src_y_accum; uint32_t x_frac, nx_frac, y_frac, ny_frac;
  int x, y, ty;
  if (iBpp != 8 && iBpp != 16) return;
  src_y_accum = FRAC_VAL/2;
  const uint32_t src_x_frac = (srcWidth * FRAC_VAL) / dstWidth;
  const uint32_t src_y_frac = (srcHeight * FRAC_VAL) / dstHeight;
  const uint32_t r_mask = 0xf800f800, g_mask = 0x07e007e0, b_mask = 0x001f001f;
  uint8_t *s, *d; uint16_t *s16, *d16; uint32_t x_frac2, y_frac2;

  for (y=0; y<dstHeight; y++) {
    ty = src_y_accum >> FRAC_BITS;
    y_frac = src_y_accum & FRAC_MASK;
    src_y_accum += src_y_frac;
    ny_frac = FRAC_VAL - y_frac;
    y_frac2 = ny_frac | (y_frac << 16);
    s = &srcImage[ty * srcWidth];
    s16 = (uint16_t *)&srcImage[ty * srcWidth * 2];
    d = &dstImage[y * dstWidth];
    d16 = (uint16_t *)&dstImage[y * dstWidth * 2];
    src_x_accum = FRAC_VAL/2;

    if (iBpp == 8) {
      for (x=0; x<dstWidth; x++) {
        uint32_t tx = src_x_accum >> FRAC_BITS;
        x_frac = src_x_accum & FRAC_MASK; nx_frac = FRAC_VAL - x_frac; x_frac2 = nx_frac | (x_frac << 16);
        src_x_accum += src_x_frac;
        uint32_t p00 = s[tx], p10 = s[tx+1], p01 = s[tx+srcWidth], p11 = s[tx+srcWidth+1];
#ifdef __ARM_FEATURE_SIMD32
        p00 = __SMLAD(p00 | (p10<<16), x_frac2, FRAC_VAL/2) >> FRAC_BITS;
        p01 = __SMLAD(p01 | (p11<<16), x_frac2, FRAC_VAL/2) >> FRAC_BITS;
        p00 = __SMLAD(p00 | (p01<<16), y_frac2, FRAC_VAL/2) >> FRAC_BITS;
#else
        p00 = ((p00*nx_frac)+(p10*x_frac)+FRAC_VAL/2)>>FRAC_BITS;
        p01 = ((p01*nx_frac)+(p11*x_frac)+FRAC_VAL/2)>>FRAC_BITS;
        p00 = ((p00*ny_frac)+(p01*y_frac)+FRAC_VAL/2)>>FRAC_BITS;
#endif
        *d++ = (uint8_t)p00;
      }
    } else {
      for (x=0; x<dstWidth; x++) {
        uint32_t tx = src_x_accum >> FRAC_BITS;
        x_frac = src_x_accum & FRAC_MASK; nx_frac = FRAC_VAL - x_frac; x_frac2 = nx_frac | (x_frac << 16);
        src_x_accum += src_x_frac;
        uint16_t *S = s16 + tx;
        uint32_t p00 = __builtin_bswap16(S[0]), p10 = __builtin_bswap16(S[1]),
                 p01 = __builtin_bswap16(S[srcWidth]), p11 = __builtin_bswap16(S[srcWidth+1]);
#ifdef __ARM_FEATURE_SIMD32
        uint32_t r00, r01, g00, g01, b00, b01;
        {
          uint32_t P0 = p00 | (p10<<16), P1 = p01 | (p11<<16);
          r00 = (P0 & r_mask) >> 1; g00 = P0 & g_mask; b00 = P0 & b_mask;
          r01 = (P1 & r_mask) >> 1; g01 = P1 & g_mask; b01 = P1 & b_mask;
          r00 = __SMLAD(r00, x_frac2, FRAC_VAL/2) >> FRAC_BITS;
          r01 = __SMLAD(r01, x_frac2, FRAC_VAL/2) >> FRAC_BITS;
          r00 = __SMLAD(r00 | (r01<<16), y_frac2, FRAC_VAL/2) >> FRAC_BITS;
          g00 = __SMLAD(g00, x_frac2, FRAC_VAL/2) >> FRAC_BITS;
          g01 = __SMLAD(g01, x_frac2, FRAC_VAL/2) >> FRAC_BITS;
          g00 = __SMLAD(g00 | (g01<<16), y_frac2, FRAC_VAL/2) >> FRAC_BITS;
          b00 = __SMLAD(b00, x_frac2, FRAC_VAL/2) >> FRAC_BITS;
          b01 = __SMLAD(b01, x_frac2, FRAC_VAL/2) >> FRAC_BITS;
          b00 = __SMLAD(b00 | (b01<<16), y_frac2, FRAC_VAL/2) >> FRAC_BITS;
        }
#else
        uint32_t r00 = (p00 & r_mask) >> 1, g00 = p00 & g_mask, b00 = p00 & b_mask;
        uint32_t r10 = (p10 & r_mask) >> 1, g10 = p10 & g_mask, b10 = p10 & b_mask;
        uint32_t r01 = (p01 & r_mask) >> 1, g01 = p01 & g_mask, b01 = p01 & b_mask;
        uint32_t r11 = (p11 & r_mask) >> 1, g11 = p11 & g_mask, b11 = p11 & b_mask;
        r00 = ((r00*nx_frac)+(r10*x_frac)+FRAC_VAL/2)>>FRAC_BITS;
        r01 = ((r01*nx_frac)+(r11*x_frac)+FRAC_VAL/2)>>FRAC_BITS;
        r00 = ((r00*ny_frac)+(r01*y_frac)+FRAC_VAL/2)>>FRAC_BITS;
        g00 = ((g00*nx_frac)+(g10*x_frac)+FRAC_VAL/2)>>FRAC_BITS;
        g01 = ((g01*nx_frac)+(g11*x_frac)+FRAC_VAL/2)>>FRAC_BITS;
        g00 = ((g00*ny_frac)+(g01*y_frac)+FRAC_VAL/2)>>FRAC_BITS;
        b00 = ((b00*nx_frac)+(b10*x_frac)+FRAC_VAL/2)>>FRAC_BITS;
        b01 = ((b01*nx_frac)+(b11*x_frac)+FRAC_VAL/2)>>FRAC_BITS;
        b00 = ((b00*ny_frac)+(b01*y_frac)+FRAC_VAL/2)>>FRAC_BITS;
#endif
        uint32_t P = ((r00<<1)&r_mask) | (g00&g_mask) | (b00&b_mask);
        *d16++ = (uint16_t)__builtin_bswap16(P);
      }
    }
  }
}

void cropImage(int srcWidth, int srcHeight, uint8_t *srcImage,
               int startX, int startY, int dstWidth, int dstHeight,
               uint8_t *dstImage, int iBpp)
{
  uint32_t *s32, *d32; int x, y;
  if (startX < 0 || startY < 0 || (startX+dstWidth) > srcWidth || (startY+dstHeight) > srcHeight) return;
  if (iBpp != 8 && iBpp != 16) return;

  if (iBpp == 8) {
    uint8_t *s, *d;
    for (y=0; y<dstHeight; y++) {
      s = &srcImage[srcWidth * (y + startY) + startX];
      d = &dstImage[(dstWidth * y)];
      x = 0;
      if ((intptr_t)s & 3 || (intptr_t)d & 3) { for (; x<dstWidth; x++) *d++ = *s++; }
      else {
        s32 = (uint32_t *)s; d32 = (uint32_t *)d;
        for (; x<dstWidth-3; x+=4) *d32++ = *s32++;
        s = (uint8_t *)s32; d = (uint8_t *)d32;
        for (; x<dstWidth; x++) *d++ = *s++;
      }
    }
  } else {
    uint16_t *s, *d;
    for (y=0; y<dstHeight; y++) {
      s = (uint16_t *)&srcImage[2 * srcWidth * (y + startY) + startX * 2];
      d = (uint16_t *)&dstImage[(dstWidth * y * 2)];
      x = 0;
      if ((intptr_t)s & 2 || (intptr_t)d & 2) { for (; x<dstWidth; x++) *d++ = *s++; }
      else {
        s32 = (uint32_t *)s; d32 = (uint32_t *)d;
        for (; x<dstWidth-1; x+=2) *d32++ = *s32++;
        s = (uint16_t *)s32; d = (uint16_t *)d32;
        for (; x<dstWidth; x++) *d++ = *s++;
      }
    }
  }
}

/* Guard: must be camera model */
#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_CAMERA
#error "Invalid model for current sensor (need CAMERA model)"
#endif

/* ---- OV767X override bits (from your example) ---- */
#include <Arduino.h>
#include <Wire.h>
#define digitalPinToBitMask(P) (1 << (digitalPinToPinName(P) % 32))
#define portInputRegister(P)   ((P == 0) ? &NRF_P0->IN : &NRF_P1->IN)

int OV7675::begin(int resolution, int format, int fps)
{
  pinMode(OV7670_VSYNC, INPUT);
  pinMode(OV7670_HREF,  INPUT);
  pinMode(OV7670_PLK,   INPUT);
  pinMode(OV7670_XCLK,  OUTPUT);

  vsyncPort = portInputRegister(digitalPinToPort(OV7670_VSYNC)); vsyncMask = digitalPinToBitMask(OV7670_VSYNC);
  hrefPort  = portInputRegister(digitalPinToPort(OV7670_HREF));  hrefMask  = digitalPinToBitMask(OV7670_HREF);
  pclkPort  = portInputRegister(digitalPinToPort(OV7670_PLK));   pclkMask  = digitalPinToBitMask(OV7670_PLK);

  bool ret = OV767X::begin(VGA, format, fps);
  width = OV767X::width(); height = OV767X::height();
  bytes_per_pixel = OV767X::bytesPerPixel(); bytes_per_row = width * bytes_per_pixel;
  resize_height = 2;

  buf_mem = NULL; raw_buf = NULL; intrp_buf = NULL;
  return ret;
}

int OV7675::allocate_scratch_buffs()
{
  buf_rows = height / resize_row_sz * resize_height;
  buf_size = bytes_per_row * buf_rows;
  buf_mem = ei_malloc(buf_size);
  if (buf_mem == NULL) { ei_printf("failed to create buf_mem\r\n"); return false; }
  raw_buf = (uint8_t *)DWORD_ALIGN_PTR((uintptr_t)buf_mem);
  return 0;
}

int OV7675::deallocate_scratch_buffs()
{
  ei_free(buf_mem); buf_mem = NULL; return 0;
}

void OV7675::readFrame(void* buffer)
{
  allocate_scratch_buffs();
  uint8_t* out = (uint8_t*)buffer;
  noInterrupts();
  while ((*vsyncPort & vsyncMask) == 0);
  while ((*vsyncPort & vsyncMask) != 0);

  int out_row = 0;
  for (int raw_height = 0; raw_height < height; raw_height += buf_rows) {
    readBuf();
    resizeImage(width, buf_rows, raw_buf, resize_col_sz, resize_height, &(out[out_row]), 16);
    out_row += resize_col_sz * resize_height * bytes_per_pixel;
  }
  interrupts();
  deallocate_scratch_buffs();
}

void OV7675::readBuf()
{
  int offset = 0;
  uint32_t ulPin = 33; // P1.xx
  NRF_GPIO_Type * port = nrf_gpio_pin_port_decode(&ulPin);

  for (int i = 0; i < buf_rows; i++) {
    while ((*hrefPort & hrefMask) == 0);
    for (int col = 0; col < bytes_per_row; col++) {
      while ((*pclkPort & pclkMask) != 0);
      uint32_t in = port->IN;
      in >>= 2; in &= 0x3f03; in |= (in >> 6);
      raw_buf[offset++] = in;
      while ((*pclkPort & pclkMask) == 0);
    }
    while ((*hrefPort & hrefMask) != 0);
  }
}
