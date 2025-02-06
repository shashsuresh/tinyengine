/* ----------------------------------------------------------------------
 * Project: Tiny Training Engine, MCUNetV3
 * Title:   main.cpp
 *
 * Reference papers:
 *  - MCUNet: Tiny Deep Learning on IoT Device, NeurIPS 2020
 *  - MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NeurIPS 2021
 *  - MCUNetV3: On-Device Training Under 256KB Memory, NeurIPS 2022
 * Contact authors:
 *  - Wei-Chen Wang, wweichen@mit.edu
 *  - Wei-Ming Chen, wmchen@mit.edu
 *  - Ji Lin, jilin@mit.edu
 *  - Ligeng Zhu, ligeng@mit.edu
 *  - Song Han, songhan@mit.edu
 *  - Chuang Gan, ganchuang@csail.mit.edu
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include "main.h"
#include "camera.h"
#include "lcd.h"
#include "profile.h"
#include "stdio.h"
#include "string.h"
#include "testing_data/golden_data.h"
#include "testing_data/images.h"
extern "C" {
#include "genNN.h"
#include "tinyengine_function.h"
}
#define SHOWIMG

#include "stm32746g_discovery.h"

static void SystemClock_Config(void);
static void Error_Handler(void);
static void CPU_CACHE_Enable(void);
static void MX_GPIO_Init(void);

#define IMAGE_H 80
#define IMAGE_W 80
#define INPUT_CH 160
#define OUTPUT_CH 2
#define IMAGES 6

void SystemClock_Config(void);
void StartDefaultTask(void const *argument);

signed char out_int[OUTPUT_CH];

float labels[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
void train(int cls) {
  char cbuf[20];
  for (int i = 0; i < 10; i++) {
    if (i == cls) {
      labels[i] = 1.0f;
    } else
      labels[i] = 0.0f;
  }
  invoke(labels);
}

void invoke_new_weights_givenimg(signed char *out_int8) {
  invoke_inf();
  signed char *output = (signed char *)getOutput();
  for (int i = 0; i < 10; i++)
    out_int8[i] = output[i];
}

#define BUTTON1_Pin GPIO_PIN_0
#define BUTTON1_GPIO_Port GPIOA
#define BUTTON2_Pin GPIO_PIN_10
#define BUTTON2_GPIO_Port GPIOF

#define RES_W 128
#define RES_H 120

uint16_t *RGBbuf;
#define ENABLE_TRAIN

uint16_t exec_ctr = 0;

int main(void) {
  char buf[150];
  char showbuf[150];

  CPU_CACHE_Enable();
  HAL_Init();

  SystemClock_Config();

  MX_GPIO_Init();

  BSP_PB_Init(BUTTON_KEY, BUTTON_MODE_GPIO);

  lcdsetup();

  uint32_t start, end, starti, endi;

  int t_mode = 0;
  const signed char* train_set[] = {mcunetv3_assets_vww_noperson6_jpg, mcunetv3_assets_vww_noperson7_jpg, mcunetv3_assets_vww_person3_jpg, mcunetv3_assets_vww_person1_jpg, mcunetv3_assets_vww_noperson2_jpg, mcunetv3_assets_vww_person6_jpg, mcunetv3_assets_vww_person7_jpg, mcunetv3_assets_vww_noperson1_jpg, mcunetv3_assets_vww_person2_jpg, mcunetv3_assets_vww_noperson3_jpg};
  const signed char* test_set[] = {person, no_person};

  int test_img = 0;
  int train_img = 0;

  while (1) {

    endi = HAL_GetTick();

    uint8_t button0 = BSP_PB_GetState(BUTTON_KEY) == GPIO_PIN_SET;
    uint8_t button1 = !HAL_GPIO_ReadPin(BUTTON1_GPIO_Port, BUTTON1_Pin);
    uint8_t button2 = !HAL_GPIO_ReadPin(BUTTON2_GPIO_Port, BUTTON2_Pin);

    char s[1];
    s[0] = 'c';
    recieveChar(s);
    if (s[0] == '3') {
        t_mode = 1;
    }
    if (s[0] == '4') {
    	exec_ctr=0;
        t_mode = 0;
    }
    if (t_mode) {
    	if (button0) {
            if (train_img == sizeof(train_set)/sizeof(train_set[0]) - 1) {
            	train_img = 0;
            } else {
            	train_img++;
            }
    	}
        signed char *input = getInput();
    	for (int i = 0; i < 80 * 80 * 3; i++) {
    	    input[i] = train_set[train_img][i];  // Image of person
    	}
    	RGBbuf = (uint16_t *)&input[80 * 80 * 4];

        starti = HAL_GetTick();

        for (int i = 0; i < 80; i++) {
            for (int j = 0; j < 80; j++) {
              uint8_t red = (int32_t)input[(80 * i + j) * 3] + 128;
              uint8_t green = (int32_t)input[(80 * i + j) * 3 + 1] + 128;
              uint8_t blue = (int32_t)input[(80 * i + j) * 3 + 2] + 128;

              uint16_t b = (blue >> 3) & 0x1f;
              uint16_t g = ((green >> 2) & 0x3f) << 5;
              uint16_t r = ((red >> 3) & 0x1f) << 11;

              RGBbuf[j + 80 * i] = (uint16_t)(r | g | b);
            }
          }
        loadRGB565LCD(10, 10, 80, 80, RGBbuf, 3);
      if ((button2 || button1 || s[0] == '1' || s[0] == '2')) {
        int label = 0;
        if (button2 || s[0] == '1') {
          sprintf(showbuf, "Train cls 1");
          label = 1;
        } else {
          sprintf(showbuf, "Train cls 0");
          label = 0;
        }

        start = HAL_GetTick();
        invoke_new_weights_givenimg(out_int);
        int answer_right = 0;
        int p;
        if (out_int[0] > out_int[1]) {
          p = 0;
          if (label == 1)
            answer_right = 0;
          else
            answer_right = 1;
        } else {
          p = 1;
          if (label == 1)
            answer_right = 1;
          else
            answer_right = 0;
        }
        end = HAL_GetTick();
        detectResponse(answer_right, 0, t_mode, p, label);

        displaystring(showbuf, 273, 10);
        start = HAL_GetTick();
        train(label);
        end = HAL_GetTick();
        sprintf(showbuf, "Train done ");
        displaystring(showbuf, 273, 10);
        detectResponse(answer_right, end - start, t_mode, p, label);
      }
    } else {
    	if (button0) {
            if (test_img == sizeof(test_set)/sizeof(test_set[0]) - 1) {
            	test_img = 0;
            } else {
            	test_img++;
            }
    	}
        signed char *input = getInput();
    	for (int i = 0; i < 80 * 80 * 3; i++) {
    	    input[i] = test_set[test_img][i];  // Image of person
    	}
    	RGBbuf = (uint16_t *)&input[80 * 80 * 4];

        starti = HAL_GetTick();

        for (int i = 0; i < 80; i++) {
            for (int j = 0; j < 80; j++) {
              uint8_t red = (int32_t)input[(80 * i + j) * 3] + 128;
              uint8_t green = (int32_t)input[(80 * i + j) * 3 + 1] + 128;
              uint8_t blue = (int32_t)input[(80 * i + j) * 3 + 2] + 128;

              uint16_t b = (blue >> 3) & 0x1f;
              uint16_t g = ((green >> 2) & 0x3f) << 5;
              uint16_t r = ((red >> 3) & 0x1f) << 11;

              RGBbuf[j + 80 * i] = (uint16_t)(r | g | b);
            }
          }
        loadRGB565LCD(10, 10, 80, 80, RGBbuf, 3);

      start = HAL_GetTick();
      invoke_new_weights_givenimg(out_int);
      int person = 0;
      if (out_int[0] > out_int[1]) {
        person = 1;
      } else {
        person = 0;
      }
      end = HAL_GetTick();
      sprintf(showbuf, " Inf %d    ", exec_ctr);
      displaystring(showbuf, 273, 10);
      detectResponse(person, end - starti, t_mode, 0, 0);
    }
    exec_ctr++;
  }
}
void SystemClock_Config(void) {
  RCC_ClkInitTypeDef RCC_ClkInitStruct;
  RCC_OscInitTypeDef RCC_OscInitStruct;
  HAL_StatusTypeDef ret = HAL_OK;

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 25;
  RCC_OscInitStruct.PLL.PLLN = 432;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 9;

  ret = HAL_RCC_OscConfig(&RCC_OscInitStruct);
  if (ret != HAL_OK) {
    while (1) {
      ;
    }
  }

  ret = HAL_PWREx_EnableOverDrive();
  if (ret != HAL_OK) {
    while (1) {
      ;
    }
  }

  RCC_ClkInitStruct.ClockType = (RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK |
                                 RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2);
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  ret = HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_7);
  if (ret != HAL_OK) {
    while (1) {
      ;
    }
  }
}
static void Error_Handler(void) {

  BSP_LED_On(LED1);
  while (1) {
  }
}

static void CPU_CACHE_Enable(void) {

  SCB_EnableICache();

  SCB_EnableDCache();
}

static void MX_GPIO_Init(void) {
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOG_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOJ_CLK_ENABLE();
  __HAL_RCC_GPIOI_CLK_ENABLE();
  __HAL_RCC_GPIOK_CLK_ENABLE();
  __HAL_RCC_GPIOF_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();

  HAL_GPIO_WritePin(OTG_FS_PowerSwitchOn_GPIO_Port, OTG_FS_PowerSwitchOn_Pin,
                    GPIO_PIN_SET);

  HAL_GPIO_WritePin(GPIOI, ARDUINO_D7_Pin | ARDUINO_D8_Pin, GPIO_PIN_RESET);

  HAL_GPIO_WritePin(LCD_BL_CTRL_GPIO_Port, LCD_BL_CTRL_Pin, GPIO_PIN_SET);

  HAL_GPIO_WritePin(LCD_DISP_GPIO_Port, LCD_DISP_Pin, GPIO_PIN_SET);

  HAL_GPIO_WritePin(DCMI_PWR_EN_GPIO_Port, DCMI_PWR_EN_Pin, GPIO_PIN_RESET);

  HAL_GPIO_WritePin(GPIOG, ARDUINO_D4_Pin | ARDUINO_D2_Pin | EXT_RST_Pin,
                    GPIO_PIN_RESET);

  GPIO_InitStruct.Pin = OTG_HS_OverCurrent_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(OTG_HS_OverCurrent_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = RMII_TXD1_Pin | RMII_TXD0_Pin | RMII_TX_EN_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF11_ETH;
  HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = ULPI_D7_Pin | ULPI_D6_Pin | ULPI_D5_Pin | ULPI_D3_Pin |
                        ULPI_D2_Pin | ULPI_D1_Pin | ULPI_D4_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF10_OTG_HS;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = SPDIF_RX0_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF8_SPDIFRX;
  HAL_GPIO_Init(SPDIF_RX0_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = OTG_FS_VBUS_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(OTG_FS_VBUS_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = Audio_INT_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_EVT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(Audio_INT_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = OTG_FS_P_Pin | OTG_FS_N_Pin | OTG_FS_ID_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF10_OTG_FS;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = OTG_FS_PowerSwitchOn_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(OTG_FS_PowerSwitchOn_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = ARDUINO_D7_Pin | ARDUINO_D8_Pin | LCD_DISP_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOI, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = uSD_Detect_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(uSD_Detect_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = LCD_BL_CTRL_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LCD_BL_CTRL_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = OTG_FS_OverCurrent_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(OTG_FS_OverCurrent_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = TP3_Pin | NC2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOH, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = ARDUINO_SCK_D13_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF5_SPI2;
  HAL_GPIO_Init(ARDUINO_SCK_D13_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = DCMI_PWR_EN_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(DCMI_PWR_EN_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = GPIO_PIN_11;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOI, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = LCD_INT_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_EVT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(LCD_INT_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = ULPI_NXT_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF10_OTG_HS;
  HAL_GPIO_Init(ULPI_NXT_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = ARDUINO_D4_Pin | ARDUINO_D2_Pin | EXT_RST_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = ULPI_STP_Pin | ULPI_DIR_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF10_OTG_HS;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = RMII_MDC_Pin | RMII_RXD0_Pin | RMII_RXD1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF11_ETH;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = RMII_RXER_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(RMII_RXER_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = RMII_REF_CLK_Pin | RMII_MDIO_Pin | RMII_CRS_DV_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF11_ETH;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = ULPI_CLK_Pin | ULPI_D0_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF10_OTG_HS;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = ARDUINO_MISO_D12_Pin | ARDUINO_MOSI_PWM_D11_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF5_SPI2;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = BUTTON1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(BUTTON1_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = BUTTON2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(BUTTON2_GPIO_Port, &GPIO_InitStruct);
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line) {

  while (1) {
  }
}
#endif

