#!/bin/sh -f

../../ninjatools/plot_kikusui_powersupply.py \
	-i data/210405_WERC_RadTest_EnotoLabPowerSupply_UnixTime.dat \
	-c 2 --title "WERC radiation test, TAC-DAQ power supply" \
	--outbasename "out/210405_WERC_RadTest_EnotoLabPowerSupply_tacdaq_all"

../../ninjatools/plot_kikusui_powersupply.py \
	-i data/210405_WERC_RadTest_EnotoLabPowerSupply_UnixTime.dat \
	-c 2 --title "WERC radiation test, TAC-DAQ power supply" \
	--outbasename "out/210405_WERC_RadTest_EnotoLabPowerSupply_tacdaq_beam1" \
	--start "2021-04-05T13:50:00" --stop "2021-04-05T17:00:00"
