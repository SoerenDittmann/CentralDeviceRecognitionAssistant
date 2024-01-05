from pickle import dump
from data_handling.utils import connect, load_select_pad


# Connection parameters
param_dic = {
    "host"      : "localhost",
    "database"  : "sensordata",
    "user"      : "",
    "password"  : ""
}

# connect to the database
conn = connect(param_dic)

#Create dictionary to store which dataset belongs to which sensortype
sensor_dic = {
    "vibration_sensor" : {},
    "acceleration_sensor" : {},
    "velocity_sensor" : {},
    "position_sensor" : {},
    "temperature_sensor" : {},
    "gyroscope" : {},
    "voltage_sensor" : {},
    "current_sensor" : {},
    "power_sensor" : {},
    "pressure_sensor" : {},
    "humidity_sensor" : {},
    "volume_flow_sensor" : {},
    "torque_sensor" : {},
    "force_sensor" : {}
    }

# ## CaseWesternReserve_BearingDataCenter
# 
# ### 1. Dataset - Baseline (cwr_base4hp)
# 
# Properly working bearing<br>
# Motor load = 4 HP<br>
# Motor speed = 1730rpm<br>
# Sample frequency = 12kHz<br>

cwr_base4hp1730rpm = load_select_pad(conn, "cwr_base4hp1730rpm")
sensor_dic["vibration_sensor"]["cwr_base4hp1730rpm"] = cwr_base4hp1730rpm


# ### 2. Dataset - Baseline (cwr_default0hp1797rpm)###
# 
# Faulty bearing<br>
# Motor load = 0 HP<br>
# Motor speed = 1797rpm<br>
# Sample frequency = 48kHz<br>
# Damaged bearing = Drive end bearing<br>
# Fault diameter = 0.021"<br>
# Fault position = Outer Race @12:00 Opposite<br>
cwr_default0hp1797rpm = load_select_pad(conn, "cwr_default0hp1797rpm")
sensor_dic["vibration_sensor"]["cwr_default0hp1797rpm"] = cwr_default0hp1797rpm


# ### 3. Dataset - Baseline (cwr_fefault0hp1797rpm)
# 
# Properly working bearing<br>
# Motor load = 0 HP<br>
# Motor speed = 1797rpm<br>
# Sample frequency = 12kHz<br>
# Damaged bearing = Fan end bearing<br>
# Fault diameter = 0.021"<br>
# Fault position = Outer Race @12:00 Opposite<br>
cwr_fefault0hp1797rpm = load_select_pad(conn, "cwr_fefault0hp1797rpm")
sensor_dic["vibration_sensor"]["cwr_fefault0hp1797rpm"] = cwr_fefault0hp1797rpm


# ## FemtoBearing DataSet
# 
# ### 1. Dataset - Vibration sensor (femto_vib_c1)
# 
# 17 different bearings operated in 3 different conditions (7/7/3)<br>
# 2560 samples (25.6kHz) are recorded each 10 seconds<br>
# Condition 1 (1800 rpm, 4000 N)<br>
# **Timestamps available**
femto_vib_c1 = load_select_pad(conn, "femto_vib_c1")
femto_vib_c1 = femto_vib_c1.iloc[:,4:]
#--------Add to data dictionary----------------------
sensor_dic["vibration_sensor"]["femto_vib_c1"] = femto_vib_c1


# ### 2. Dataset - Temperature sensor (femto_temp_c1)
# 
# 17 different bearings operated in 3 different conditions (7/7/3)<br>
# 600 samples (0.1 Hz) are recorded each 10 seconds<br>
# Condition 1 (1800 rpm, 4000 N)<br>
# **Timestamps available**
femto_temp_c1 = load_select_pad(conn, "femto_temp_c1")
femto_temp_c1 = femto_temp_c1.iloc[:,4:] #leave out time stamps
sensor_dic["temperature_sensor"]["femto_temp_c1"] = femto_temp_c1


# ## IMS Bearing DataSet
# 
# ### 1. Dataset - Vibration sensor (ims_bearing)
# 
# Each data set consists of individual files that are 1-second vibration signal snapshots of 8 sensors each recorded at specific intervals. Each file consists of 20,480 points with the sampling rate set at **20 kHz, 2000rpm**<br>
# File Name: 2003.11.01.18.01<br>
# No timestamps available
ims_bearing = load_select_pad(conn, "ims_bearing")
sensor_dic["vibration_sensor"]["ims_bearing"] = ims_bearing


# ## SensiML DataSet
# 
# ### 1. Dataset - Vibration sensor (sensiml)
# 
# Each data set consists of individual files that are 1-second vibration signal snapshots of 8 sensors each recorded at specific intervals. Each file consists of 20,480 points with the sampling rate set at **20 kHz, 2000rpm**<br>
# File Name: 2003.11.01.18.01<br>
# No timestamps available
sensiml = load_select_pad(conn, "sensiml")
sensor_dic["vibration_sensor"]["sensiml"] = sensiml.iloc[:,:3]
sensor_dic["gyroscope"]["sensiml"] = sensiml.iloc[:,3:]


# ## CNC MillToolWear DataSet
# 
# ### 1. Dataset - Various sensors (cnc_mill_tool_wear)
# 
# Positon, Velocity, Acceleration (x,y,z-axis) + actual vs. command) + current von 4 motors.<br>
# Random Selection. Just thought experiment 1 looked best, because it is the most normal.<br>
# Samplefrequenz = 10Hz<br>
# **Conditions**<br>
# material = wax<br>
# feedrate = 6<br>
# clamp_pressure = 4<br>
# tool_condition = unworn<br>
# machining_finalized = yes<br>
# passed_visual_inspection = yes<br>
# No timestamps available
cnc_mill_tool_wear_exp1 = load_select_pad(conn, "cnc_mill_tool_wear")

#>>z-axis current feedback, dcbus voltage, output voltage, output current<<  are all zeroes throughout the entire dataset

#---------Build subsets of data--------------------
cnc_mill_tool_wear_exp1_acc = cnc_mill_tool_wear_exp1[["x1_actualacceleration","y1_actualacceleration","z1_actualacceleration","s1_actualacceleration"]]
cnc_mill_tool_wear_exp1_vel = cnc_mill_tool_wear_exp1[["x1_actualvelocity","y1_actualvelocity","z1_actualvelocity","s1_actualvelocity"]]
cnc_mill_tool_wear_exp1_pos = cnc_mill_tool_wear_exp1[["x1_actualposition","y1_actualposition","z1_actualposition","s1_actualposition"]]

cnc_mill_tool_wear_exp1_vlt = cnc_mill_tool_wear_exp1[["x1_dcbusvoltage","y1_dcbusvoltage","s1_dcbusvoltage","x1_outputvoltage","y1_outputvoltage","s1_outputvoltage"]]
cnc_mill_tool_wear_exp1_cur = cnc_mill_tool_wear_exp1[["x1_outputcurrent","y1_outputcurrent","s1_outputcurrent","x1_currentfeedback","y1_currentfeedback","s1_currentfeedback"]]
cnc_mill_tool_wear_exp1_pwr = cnc_mill_tool_wear_exp1[["x1_outputpower","y1_outputpower","s1_outputpower"]]

#--------Add to data dictionary----------------------
sensor_dic["acceleration_sensor"]["cnc_mill_tool_wear_exp1"] = cnc_mill_tool_wear_exp1_acc
sensor_dic["velocity_sensor"]["cnc_mill_tool_wear_exp1"] = cnc_mill_tool_wear_exp1_vel
sensor_dic["position_sensor"]["cnc_mill_tool_wear_exp1"] = cnc_mill_tool_wear_exp1_pos
sensor_dic["voltage_sensor"]["cnc_mill_tool_wear_exp1"] = cnc_mill_tool_wear_exp1_vlt
sensor_dic["current_sensor"]["cnc_mill_tool_wear_exp1"] = cnc_mill_tool_wear_exp1_cur
sensor_dic["power_sensor"]["cnc_mill_tool_wear_exp1"] = cnc_mill_tool_wear_exp1_pwr


# ## Gas Turbine and NOx Data Set
# 
# ### 1. Dataset - Various sensors (gasturbinenox)
# 
# dataset contains 36733 instances of 11 sensor measures aggregated over one hour (by means of average or sum). dates are not given in the instances but the data are sorted in chronological order
gasturbinenox = load_select_pad(conn, "gasturbinenox")

#---------Build subsets of data--------------------
gasturbinenox_temp = gasturbinenox["at"]
gasturbinenox_prs = gasturbinenox["ap"]
gasturbinenox_hum = gasturbinenox["ah"]

#--------Add to data dictionary----------------------
sensor_dic["temperature_sensor"]["gasturbinenox"] = gasturbinenox_temp
sensor_dic["pressure_sensor"]["gasturbinenox"] = gasturbinenox_prs
sensor_dic["humidity_sensor"]["gasturbinenox"] = gasturbinenox_hum


# ## MillingNASA
# 
# ### 1. Dataset - Various sensors (millingnasa)
# 
# **Available information**<br>
# acoustic emission sensor spindle<br>
# acoustic emission sensor table<br>
# vibration sensor spindle<br>
# vibration sensor table<br>
# spindle motor current sensor AC<br>
# spindle motor current sensor DC<br>
# No time stamp
millingnasa = load_select_pad(conn, "millingnasa")
sensor_dic["vibration_sensor"]["millingnasa"] = millingnasa[["vib_table","vib_spindle"]]
sensor_dic["current_sensor"]["millingnasa"] = millingnasa[["smcac","smcdc"]]


# ## One Year Industrial Component Degradation
# 
# ### 1. Dataset - Various sensors (OneYearIndustrialComponentDegradation)
# 
# **Available information**<br>
# acoustic emission sensor spindle<br>
# acoustic emission sensor table<br>
# vibration sensor spindle<br>
# vibration sensor table<br>
# spindle motor current sensor AC<br>
# spindle motor current sensor DC<br>
# VAX sensor = constant -> not interesting<br>
# **Timestamp available!**
OYICD = load_select_pad(conn, "industrial_component_degradation")
sensor_dic["position_sensor"]["OYICD"] = OYICD[["pcut__ctrl_position_controller__actual_position","psvolfilm__ctrl_position_controller__actual_position"]]
sensor_dic["velocity_sensor"]["OYICD"] = OYICD[["pcut__ctrl_position_controller__actual_speed","psvolfilm__ctrl_position_controller__actual_speed"]]
sensor_dic["torque_sensor"]["OYICD"] = OYICD[["pcut__motor_torque"]]


# ## ENGIE windturbine data
# 
# ### 1. Dataset - Various sensors (ENGIE_windturbine)
# 
# 
# Select only a certain wind turbine. I randomly chose R80721.<br>
# Only features like min,max,std available.<br>
# Sorted according to date_time.<br>
# Other names include = R80711, R80790, R80736
# **Timestamp available!**

#engie_windturbine = load_select_pad(conn, "engie_windturbine")
#engie_windturbine = engie_windturbine.iloc[:,2:] 

#ACHTUNG Spalten vom Type objecct dabei

#--------Add to data dictionary----------------------
#sensor_dic["vibration_sensor"]["ENGIE_windturbine"] = ENGIE_windturbine[["column1","column2"]]


# # 3. Systemslevel
# 
# ## Condition monitoring hydraulic systems
# 
# ### 1. Data sets - temperature sensors (cond_mon_ts1_df - cond_mon_ts4_df)
# 
# The data set contains raw process sensor data (i.e. without feature extraction) which are structured as matrices with the rows representing the cycles and the columns the data points within a cycle.<br>
# cycle0 - cycle1 - cycle2 - cycle3 - cycle4 - cycle5 - ...<br>
# p1<br>
# p2<br>
# p3<br>
# ...
#---------Loading and selecting data-----------------
cond_mon_ts1_df = load_select_pad(conn, "cond_mon_ts1_df")
cond_mon_ts2_df = load_select_pad(conn, "cond_mon_ts2_df")
cond_mon_ts3_df = load_select_pad(conn, "cond_mon_ts3_df")
cond_mon_ts4_df = load_select_pad(conn, "cond_mon_ts4_df")
cond_mon_fs1_df = load_select_pad(conn, "cond_mon_fs1_df")
cond_mon_fs2_df = load_select_pad(conn, "cond_mon_fs2_df")
cond_mon_eps1_df = load_select_pad(conn, "cond_mon_eps1_df")

#--------Add to data dictionary----------------------
sensor_dic["temperature_sensor"]["cond_mon_TS1"] = cond_mon_ts1_df["cycle_0"]
sensor_dic["temperature_sensor"]["cond_mon_TS2"] = cond_mon_ts2_df["cycle_0"]
sensor_dic["temperature_sensor"]["cond_mon_TS3"] = cond_mon_ts3_df["cycle_0"]
sensor_dic["temperature_sensor"]["cond_mon_TS4"] = cond_mon_ts4_df["cycle_0"]
sensor_dic["volume_flow_sensor"]["cond_mon_FS1"] = cond_mon_fs1_df["cycle_0"]
sensor_dic["volume_flow_sensor"]["cond_mon_FS2"] = cond_mon_fs2_df["cycle_0"]
sensor_dic["power_sensor"]["cond_mon_EPS1"] = cond_mon_eps1_df["cycle_0"]


# ### 2. Data sets - vibration sensors (cond_mon_vs1_df)
# 
# The data set contains raw process sensor data (i.e. without feature extraction) which are structured as matrices with the rows representing the cycles and the columns the data points within a cycle.<br>
# cycle0 - cycle1 - cycle2 - cycle3 - cycle4 - cycle5 - ...<br>
# p1<br>
# p2<br>
# p3<br>
# ...
cond_mon_vs1_df = load_select_pad(conn, "cond_mon_vs1_df")
sensor_dic["vibration_sensor"]["cond_mon_VS1"] = cond_mon_vs1_df["cycle_0"]


# ## Genesis demonstrator data
# 
# ### 1. Data sets - acceleration sensors (genesis_demonstrator)
# 
# Both data sets contain 16220 observations taken every 50ms through an OPC DA server.
genesis_demonstrator = load_select_pad(conn, "genesis_demonstrator")

sensor_dic["acceleration_sensor"]["genesis_demonstrator"] = genesis_demonstrator["motordata_isacceleration"]
sensor_dic["velocity_sensor"]["genesis_demonstrator"] = genesis_demonstrator["motordata_actspeed"]
sensor_dic["current_sensor"]["genesis_demonstrator"] = genesis_demonstrator["motordata_actcurrent"]
sensor_dic["force_sensor"]["genesis_demonstrator"] = genesis_demonstrator["motordata_isforce"]


# ## High Storage System Data
# 
# ### 1. Data sets - acceleration sensors (high_storage_system)
# 
# Short conveyor belts with three induction sensors each

high_storage_system = load_select_pad(conn, "high_storage_system")

high_storage_system_pos = high_storage_system[["i_w_blo_weg","i_w_bhl_weg","i_w_bhr_weg","i_w_bru_weg","i_w_hr_weg","i_w_hl_weg"]]
high_storage_system_vol = high_storage_system[["o_w_blo_voltage","o_w_bhl_voltage","o_w_bhr_voltage","o_w_bru_voltage","o_w_hr_voltage","o_w_hl_voltage"]]
high_storage_system_pwr = high_storage_system[["o_w_blo_power","o_w_bhl_power","o_w_bhr_power","o_w_bru_power","o_w_hr_power","o_w_hl_power"]]

#--------Add to data dictionary----------------------
sensor_dic["position_sensor"]["high_storage_system"] = high_storage_system_pos
sensor_dic["voltage_sensor"]["high_storage_system"] = high_storage_system_vol
sensor_dic["power_sensor"]["high_storage_system"] = high_storage_system_pwr


# ## POT Daten
# 
# ### 1. Data sets - acceleration sensors (pot_data)
# 
pot_data = load_select_pad(conn, "pot_data")

#Acceleration and torque sensors contain all zeroes
#pot_data_acc = pot_data[["axes_axis_5___be___phi_actacc","axes_axis_9___be___rs2o_toplc_actacc","axes_axis_6___be___y_actacc"]]
pot_data_vel = pot_data[["axes_axis_5___be___phi_actvelo","axes_axis_9___be___rs2o_actvelo","axes_axis_6___be___y_actvelo"]]
pot_data_pos = pot_data[["axes_axis_5___be___phi_actpos","axes_axis_9___be___rs2o_actpos","axes_axis_6___be___y_actpos"]]
#pot_data_trq = pot_data[["axes_axis_5___be___phi_acttorque","axes_axis_9___be___rs2o_toplc_acttorque","axes_axis_6___be___y_acttorque"]]

#--------Add to data dictionary----------------------
#sensor_dic["acceleration_sensor"]["pot_data"] = pot_data_acc
sensor_dic["velocity_sensor"]["pot_data"] = pot_data_vel
sensor_dic["position_sensor"]["pot_data"] = pot_data_pos
#sensor_dic["torque_sensor"]["pot_data"] = pot_data_trq


# ## HMI Demonstrator
# 
# ### 1. Data sets - Position sensors (hmi_demonstrator)
# 

hmi_demonstrator = load_select_pad(conn, "hmi_demonstrator")


#--------Add to data dictionary----------------------
sensor_dic["position_sensor"]["hmi_demonstrator"] = hmi_demonstrator["actpos"]


  
with open('sensor_dic_normalized.pkl', 'wb') as filehandle:
    dump(sensor_dic, filehandle)

