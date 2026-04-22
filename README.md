# Learning

杩欎釜鐩綍鏀剧殑鏄畫宸帶鍒惰缁冧唬鐮併€傚綋鍓嶇洰鏍囦笉鏄畬鏁村鑸紝鑰屾槸鎶婁綆灞傛帶鍒跺仛绋筹細

```text
final_command = base_controller_command + neural_residual
```

褰撳墠瀹炵幇閲岋細

- 娣卞害杞达細`u_total = u_base + u_residual`
- 鍓嶈繘杞达細`forward_cmd_total = forward_cmd_base + forward_cmd_residual`
- 杞悜杞达細`yaw_cmd_total = yaw_cmd_base + yaw_cmd_residual`

涔熷氨鏄锛屼紶缁熸帶鍒跺櫒鍏堜繚璇佺郴缁熷彲鎺э紝绁炵粡缃戠粶鍙涔犫€滆ˉ澶氬皯鈥濄€?
## 闃呰璺嚎

寤鸿鎸夎繖涓『搴忕湅锛?
1. [data.py](data.py)
   鐪?CSV 杈撳叆濂戠害銆佹牱鏈繃婊ゃ€佹粦鍔ㄧ獥鍙ｃ€乼arget 瑙ｆ瀽銆?2. [train_residual.py](train_residual.py)
   鐪嬪崟杞存繁搴?residual 鎬庝箞璁粌銆?3. [train_axis_models.py](train_axis_models.py)
   鐪?depth / forward / yaw 涓夎酱鎬庝箞鍒嗗埆璁粌銆?4. [model.py](model.py)
   鐪嬪皬鍨?MLP 鐨勭粨鏋勫拰璁粌鏂瑰紡銆?5. [export_to_esp32.py](export_to_esp32.py)
   鐪嬭缁冨悗濡備綍瀵煎嚭缁?ESP32銆?
## 褰撳墠浠ｇ爜缁撴瀯

| 鏂囦欢 | 浣滅敤 |
| --- | --- |
| [data.py](data.py) | 璇诲彇 CSV銆佽繃婊ゆ棤鏁堝抚銆佹瀯閫犳粦鍔ㄧ獥鍙ｆ牱鏈€佽В鏋?residual target銆?|
| [model.py](model.py) | 绾?Python 鐨勫皬鍨?MLP 鍥炲綊鍣ㄣ€?|
| [train_residual.py](train_residual.py) | 鍗曡酱娣卞害 residual 璁粌鍏ュ彛銆?|
| [train_axis_models.py](train_axis_models.py) | 涓夎酱 residual 璁粌鍏ュ彛銆?|
| [export_to_esp32.py](export_to_esp32.py) | 鎶婅缁冨悗鐨勬ā鍨?bundle 瀵煎嚭鎴?ESP32 澶存枃浠躲€?|
| [tests/](tests) | `learning` 鐩綍鑷繁鐨勫崟鍏冩祴璇曘€?|

## 杈撳叆鍙傛暟濂戠害

杩欓噷鐨勨€滆緭鍏ュ弬鏁扳€濅富瑕佹寚涓ょ被涓滆タ锛?
- 璁粌鑴氭湰鐨?CLI 鍙傛暟
- 璁粌鎵€渚濊禆鐨?CSV 鍒楀悕鍜?target 瑙勫垯

涓嬮潰杩欎簺鍐呭鏄綋鍓嶄唬鐮佸凡缁忓疄闄呮墽琛岀殑濂戠害锛屼笉鍙槸鏂囨。璇存槑銆?
### 0. 姝ｅ紡 `sensors.csv` 琛ㄥご

濡傛灉浣犺璁?ESP32 鏃ュ織鍜?`learning/` 鐩綍瀹屽叏瀵归綈锛屽綋鍓嶆帹鑽愬浐瀹氫娇鐢ㄤ笅闈㈣繖涓€缁勫垪锛屽苟涓旀寜杩欎釜椤哄簭杈撳嚭锛?
```text
session_id
timestamp_ms
dt_ms
robot_mode
control_mode
depth_valid
imu_valid
battery_v
target_depth_cm
filtered_depth_cm
depth_speed_cm_s
depth_accel_cm_s2
roll_deg
pitch_deg
gyro_x_deg_s
gyro_y_deg_s
gyro_z_deg_s
front_distance_cm
left_distance_cm
right_distance_cm
depth_err_cm
u_base
u_residual
u_total
forward_cmd_base
forward_cmd_residual
forward_cmd_total
forward_phase_interval_ms
yaw_cmd_base
yaw_cmd_residual
yaw_cmd_total
buoyancy_dir_applied
buoyancy_pwm_applied
actuator_mask
balancing
emergency_stop
```

璇存槑锛?
- `session_id` 鐜板湪搴旇涓庡綋鍓?SD session 鏂囦欢澶瑰悕涓€鑷?- `dt_ms`銆乣robot_mode`銆乣target_depth_cm`銆乣filtered_depth_cm` 杩欎簺鍒楃洰鍓嶄富瑕佺敤浜庢棩蹇楄拷婧紝涓嶆槸褰撳墠璁粌鐗瑰緛
- `imu_valid=0` 鎴?IMU 鏁板€兼棤鏁堢殑甯э紝璁粌鏃朵細琚繃婊ゆ帀

### 1. 閫氱敤鏍锋湰杩囨护瀛楁

褰撹繖浜涘垪瀛樺湪鏃讹紝`data.py` 浼氱敤瀹冧滑杩囨护璁粌鏍锋湰锛?
- `session_id`
- `timestamp_ms`
- `control_mode`
- `depth_valid`
- `imu_valid`
- `balancing`
- `emergency_stop`

杩囨护瑙勫垯锛?
- `depth_valid=0` 鐨勫抚浼氳涓㈡帀
- `imu_valid=0` 鐨勫抚浼氳涓㈡帀
- `balancing=1` 鐨勫抚浼氳涓㈡帀
- `emergency_stop=1` 鐨勫抚浼氳涓㈡帀
- `L50` 杩欑被楂樺眰鐩爣鍛戒护浜х敓鐨勯棴鐜墽琛屽抚浼氳淇濈暀锛屼笉鍐嶅洜涓?`control_mode=0` 鎴?`manual` 鑰屼涪鎺?- 鍓嶈繘 / 杞悜杞翠换鍔″抚鍚屾牱浼氳淇濈暀
- 鍙湁鐪熸鐨?`j/k` 娴姏鐩存帶甯т細琚涪鎺夛細
  `buoyancy_dir_applied in {1,2}` 涓?`buoyancy_pwm_applied` 鎺ヨ繎 `255`锛?  鍚屾椂 `u_base=0`銆乣u_residual=0`銆乣|u_total|=100`
- 鏃堕棿鎴冲€掗€€鎴栫浉閭诲抚鏃堕棿宸繃澶ф椂锛屼細琚垏鎴愪笉鍚屽簭鍒?
### 2. 鍗曡酱娣卞害璁粌鐨勯粯璁よ緭鍏ョ壒寰?
`train_residual.py` 榛樿浣跨敤杩欎簺鍒楋細

- `depth_err_cm`
- `depth_speed_cm_s`
- `depth_accel_cm_s2`
- `roll_deg`
- `pitch_deg`
- `gyro_x_deg_s`
- `gyro_y_deg_s`
- `gyro_z_deg_s`
- `battery_v`
- `buoyancy_pwm_applied`

### 3. 涓夎酱璁粌鐨勯粯璁よ緭鍏ョ壒寰?
`train_axis_models.py` 榛樿浣跨敤杩欎簺鍏变韩鐗瑰緛锛?
- `depth_err_cm`
- `depth_speed_cm_s`
- `depth_accel_cm_s2`
- `roll_deg`
- `pitch_deg`
- `gyro_x_deg_s`
- `gyro_y_deg_s`
- `gyro_z_deg_s`
- `front_distance_cm`
- `left_distance_cm`
- `right_distance_cm`
- `battery_v`
- `u_base`
- `forward_cmd_base`
- `forward_phase_interval_ms`
- `yaw_cmd_base`

### 4. target 瑙ｆ瀽瑙勫垯

#### 鍗曡酱娣卞害璁粌鐨勮嚜鍔?target 瑙ｆ瀽椤哄簭

`train_residual.py` 鍦ㄦ湭鏄惧紡浼?`--target-column` 鏃讹紝浼氭寜杩欎釜椤哄簭鎵?target锛?
1. `residual_target_pwm`
2. `u_residual`
3. `u_total - u_base`

#### 鏄惧紡 target_column 鐨勫洖閫€瑙勫垯

褰撳墠浠ｇ爜宸茬粡鏀寔涓嬮潰杩欎簺鏄惧紡 residual 瀛楁鍦ㄥ垪缂哄け鏃惰嚜鍔ㄥ洖閫€鍒?`total - base`锛?
- `residual_target_pwm` -> `u_total - u_base`
- `u_residual` -> `u_total - u_base`
- `forward_cmd_residual` -> `forward_cmd_total - forward_cmd_base`
- `yaw_cmd_residual` -> `yaw_cmd_total - yaw_cmd_base`

杩欐剰鍛崇潃锛?
- 濡傛灉浣犺缁?`forward_cmd_residual`
- 浣嗘棩蹇楅噷娌℃湁杩欏垪
- 鍙鏈?`forward_cmd_total` 鍜?`forward_cmd_base`

浠ｇ爜鐜板湪涔熶細鎸?README 閲岀殑瑙勫垯鑷姩鎭㈠杩欎釜 residual target銆?
#### 涓夎酱璁粌鐨勯粯璁?target

`train_axis_models.py` 榛樿浣跨敤锛?
- `depth` -> `u_residual`
- `forward` -> `forward_cmd_residual`
- `yaw` -> `yaw_cmd_residual`

## 璁粌鍛戒护

涓嬮潰鍛戒护榛樿閮藉湪浠撳簱鏍圭洰褰曡繍琛屻€?
### 璁粌鍗曡酱娣卞害 residual

```powershell
python learning\train_residual.py `
  --csv D:\path\to\control_telemetry.csv `
  --output learning\artifacts\residual_model.json
```

甯哥敤鍙傛暟锛?
- `--window-size`
- `--feature-columns`
- `--target-column`
- `--hidden-dims`
- `--epochs`
- `--learning-rate`
- `--l2`
- `--val-fraction`
- `--max-dt-ms`
- `--seed`
- `--print-every`

绀轰緥锛?
```powershell
python learning\train_residual.py `
  --csv D:\path\to\control_telemetry.csv `
  --output learning\artifacts\residual_model.json `
  --window-size 5 `
  --hidden-dims 24 12 `
  --epochs 400 `
  --learning-rate 0.008 `
  --target-column residual_target_pwm
```

### 璁粌涓夎酱 residual

```powershell
python learning\train_axis_models.py `
  --csv D:\path\to\control_telemetry.csv `
  --output-dir learning\artifacts\axis_models
```

甯哥敤鍙傛暟锛?
- `--feature-columns`
- `--window-size`
- `--hidden-dims`
- `--epochs`
- `--learning-rate`
- `--l2`
- `--val-fraction`
- `--max-dt-ms`
- `--seed`
- `--print-every`
- `--depth-target-column`
- `--forward-target-column`
- `--yaw-target-column`

## 杈撳嚭鏂囦欢

璁粌鑴氭湰杈撳嚭鐨勬槸 JSON bundle锛屼笉鏄８鏉冮噸銆俠undle 閲屽寘鍚細

- 妯″瀷缁撴瀯
- 妯″瀷鍙傛暟
- 杈撳叆鏍囧噯鍖栧弬鏁?- target 鏍囧噯鍖栧弬鏁?- 鐗瑰緛椤哄簭
- 璁粌鍏冩暟鎹?
杩欐牱鍋氱殑鐩殑锛屾槸璁╄缁冨拰鏉跨鎺ㄧ悊涔嬮棿鐨勮緭鍏ラ『搴忔樉寮忓浐瀹氫笅鏉ャ€?
## 瀵煎嚭鍒?ESP32

褰撳墠瀵煎嚭宸ュ叿 [export_to_esp32.py](export_to_esp32.py) 浠嶇劧鍙湇鍔′簬娣卞害 residual 鐨勬澘绔儴缃层€?
瀵煎嚭鍛戒护锛?
```powershell
python learning\export_to_esp32.py `
  --model learning\artifacts\residual_model.json `
  --output ESP32\ResidualModelData.h
```

褰撳墠瀵煎嚭鍣ㄤ細寮虹害鏉熶笅闈㈣繖缁?feature 椤哄簭锛?
- `depth_err_cm`
- `depth_speed_cm_s`
- `depth_accel_cm_s2`
- `roll_deg`
- `pitch_deg`
- `gyro_x_deg_s`
- `gyro_y_deg_s`
- `gyro_z_deg_s`
- `battery_v`
- `buoyancy_pwm_applied`

骞朵笖褰撳墠瀵煎嚭鍣ㄥ彧鏀寔锛?
- 涓ゅ眰闅愯棌灞?- 涓€涓爣閲忚緭鍑?- 涓庢繁搴?residual 鏉跨鎺ㄧ悊濂戠害瀹屽叏涓€鑷寸殑 feature order

鎵€浠ョ幇鍦ㄧ殑鐪熷疄鐘舵€佹槸锛?
- depth residual: 璁粌 + 瀵煎嚭 + 鏉跨鎺ㄧ悊锛屾暣鏉￠摼璺凡缁忔墦閫?- forward residual: 鍙畬鎴愪簡绂荤嚎璁粌 bundle
- yaw residual: 鍙畬鎴愪簡绂荤嚎璁粌 bundle

濡傛灉浠ュ悗瑕佹妸 forward / yaw 涔熸斁鍒?ESP32 涓婅窇锛岄渶瑕佺户缁ˉ锛?
- forward/yaw 鐨勫鍑烘牸寮?- forward/yaw 鐨勬澘绔?inference wrapper
- forward/yaw 鐨勮繍琛屾椂鎺у埗鍜岄檺骞?
## 濡傛灉浠ュ悗瑕佹敼杈撳叆鍙傛暟锛岃鏀瑰摢浜涙枃浠?
### 鏀?CSV 鐗瑰緛鍒楀悕

鍏堟敼锛?
- [data.py](data.py)

鍐嶅悓姝ユ鏌ワ細

- [train_residual.py](train_residual.py)
- [train_axis_models.py](train_axis_models.py)
- [export_to_esp32.py](export_to_esp32.py)
- 杩欎釜 README

### 鏀?residual target 鍚嶇О鎴栧洖閫€瑙勫垯

鍏堟敼锛?
- [data.py](data.py)

鍐嶅悓姝ユ鏌ワ細

- [train_residual.py](train_residual.py)
- [train_axis_models.py](train_axis_models.py)
- `ESP32/SDLogger.cpp`
- 杩欎釜 README

### 鏀规澘绔帹鐞嗚緭鍏ラ『搴?
蹇呴』涓€璧风湅锛?
- [export_to_esp32.py](export_to_esp32.py)
- `ESP32/ResidualInference.cpp`
- `ESP32/ResidualModelData.h`
- 杩欎釜 README

## 娴嬭瘯

杩愯锛?
```powershell
python -m unittest discover learning\tests -v
```

褰撳墠娴嬭瘯瑕嗙洊锛?
- CSV 璇诲彇鍜岃繃婊?- 婊戝姩绐楀彛鏍锋湰鏋勯€?- 鏍囧噯鍖栧線杩?- 灏忓瀷 MLP 鐨勫熀鏈缁冭涓?- 瀵煎嚭鏍煎紡
- 涓夎酱妯″瀷瀵煎嚭
- 鏄惧紡 axis target 鐨?`total - base` 鍥為€€
