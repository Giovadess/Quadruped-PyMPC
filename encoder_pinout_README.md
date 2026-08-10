# Passive Arm Bourns Encoder Pinout

This document summarizes the wiring used for the three passive-arm Bourns absolute encoders.

The encoders use the Bourns absolute output pinout:

| Encoder pin | Signal | Meaning |
|---:|---|---|
| 1 | DI | Digital input. For single-sensor configuration, connect to GND. |
| 2 | CLK | Clock input from Arduino. |
| 3 | GND | Encoder ground. |
| 4 | DO | Digital output from encoder to Arduino. |
| 5 | VCC | Encoder supply voltage, 5 V or 3.3 V depending on encoder version. |
| 6 | CS | Chip select from Arduino. |

Current data flow:

```text
Encoder pins -> Arduino -> Serial /dev/ttyACM0 -> ROS 2 passive_arm_joint_states
```

## Confirmed Arduino read settings

The following read settings were used for the working encoder tests:

```cpp
const bool CS_ACTIVE_HIGH = false;
const bool READ_AFTER_CLK_HIGH = true;
const bool USE_TOP_10_BITS = true;
```

The encoder resolution used in the Arduino sketch is:

```cpp
const int COUNTS_PER_REV = 1024;
```

## Joint 1 encoder

Joint 1 uses Arduino digital pins **D11 to D8**.

Updated measured cable order for joint 1:

```text
1 blue
2 grey
3 black
4 white
5 pink
6 brown
```

Therefore the wiring is:

| Encoder pin | Signal | Cable color | Arduino connection |
|---:|---|---|---|
| 1 | DI | blue | D8 configured OUTPUT LOW, or directly GND |
| 2 | CLK | grey | D11 |
| 3 | GND | black | GND |
| 4 | DO | white | D10 |
| 5 | VCC | pink | 5V, or 3.3V if this is the 3.3V encoder version |
| 6 | CS | brown | D9 |

Practical connection summary:

```text
Joint 1:
blue  -> D8      // DI, driven LOW
grey  -> D11     // CLK
black -> GND
white -> D10     // DO
pink  -> 5V
brown -> D9      // CS
```

Arduino constants:

```cpp
const int J1_DI  = 8;
const int J1_CLK = 11;
const int J1_DO  = 10;
const int J1_CS  = 9;
```

## Joint 2 encoder

Joint 2 uses Arduino digital pins **D7 to D4**.

Measured cable order for joint 2:

```text
1 blue
2 grey
3 black
4 white
5 pink
6 brown
```

Therefore the wiring is:

| Encoder pin | Signal | Cable color | Arduino connection |
|---:|---|---|---|
| 1 | DI | blue | D4 configured OUTPUT LOW, or directly GND |
| 2 | CLK | grey | D7 |
| 3 | GND | black | GND |
| 4 | DO | white | D6 |
| 5 | VCC | pink | 5V, or 3.3V if this is the 3.3V encoder version |
| 6 | CS | brown | D5 |

Practical connection summary:

```text
Joint 2:
blue  -> D4      // DI, driven LOW
grey  -> D7      // CLK
black -> GND
white -> D6      // DO
pink  -> 5V
brown -> D5      // CS
```

Arduino constants:

```cpp
const int J2_DI  = 4;
const int J2_CLK = 7;
const int J2_DO  = 6;
const int J2_CS  = 5;
```

Notes:

- Joint 2 was confirmed working in the Arduino Serial Plotter.
- If possible, connect DI directly to GND instead of D4. If D4 is used, configure it as `OUTPUT LOW`.

## Joint 3 encoder

Joint 3 uses Arduino analog pins **A0 to A2** as digital pins.

Measured and confirmed working wiring:

| Encoder pin | Signal | Cable color | Arduino connection |
|---:|---|---|---|
| 1 | DI | blue | GND |
| 2 | CLK | grey | A0 |
| 3 | GND | black | GND |
| 4 | DO | white | A1 |
| 5 | VCC | pink | 5V, or 3.3V if this is the 3.3V encoder version |
| 6 | CS | brown | A2 |

Practical connection summary:

```text
Joint 3:
blue  -> GND     // DI, single-sensor mode
grey  -> A0      // CLK
black -> GND
white -> A1      // DO
pink  -> 5V
brown -> A2      // CS
```

Arduino constants:

```cpp
const int J3_CLK = A0;
const int J3_DO  = A1;
const int J3_CS  = A2;
```

Notes:

- Joint 3 was confirmed working in the Arduino Serial Plotter.
- Arduino UNO analog pins A0, A1, and A2 can be used as digital pins.

## Full Arduino pin constants

Use these constants in the three-joint Arduino sketch:

```cpp
// Joint 1
const int J1_DI  = 8;
const int J1_CLK = 11;
const int J1_DO  = 10;
const int J1_CS  = 9;

// Joint 2
const int J2_DI  = 4;
const int J2_CLK = 7;
const int J2_DO  = 6;
const int J2_CS  = 5;

// Joint 3
const int J3_CLK = A0;
const int J3_DO  = A1;
const int J3_CS  = A2;
```

For joint 1 and joint 2, if DI is connected to Arduino digital pins instead of directly to GND, drive those pins LOW in `setup()`:

```cpp
pinMode(J1_DI, OUTPUT);
digitalWrite(J1_DI, LOW);

pinMode(J2_DI, OUTPUT);
digitalWrite(J2_DI, LOW);
```

## Serial output modes

For Arduino Serial Plotter debugging, print labeled count values:

```text
joint1_count:<value>    joint2_count:<value>    joint3_count:<value>
```

For the ROS 2 Python node, stream only tab-separated radians:

```text
joint1_rad    joint2_rad    joint3_rad
```

Example:

```text
0.123456    1.234567    2.345678
```

This matches the original ROS 2 Python node format, which expects three tab-separated encoder values and publishes:

```text
arm_joint1, arm_joint2, arm_joint3
```

## Quick troubleshooting

If an encoder always reads `1023`, `0`, or does not change when rotated:

1. Power off before rewiring.
2. Re-check connector orientation. The Bourns drawing can be mirrored depending on whether you look at the socket side or cable side.
3. Verify VCC and GND first.
4. Verify CLK, DO, and CS are not swapped.
5. Make sure DI is grounded for single-sensor configuration.
6. Test one encoder at a time in the Arduino Serial Plotter.
7. If an encoder works when connected to the joint 2 plug but not on another plug, the encoder is probably fine and the issue is the connector mapping, pin assignment, or wiring path.