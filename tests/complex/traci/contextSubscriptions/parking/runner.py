#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2008-2022 German Aerospace Center (DLR) and others.
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# https://www.eclipse.org/legal/epl-2.0/
# This Source Code may also be made available under the following Secondary
# Licenses when the conditions for such availability set forth in the Eclipse
# Public License 2.0 are satisfied: GNU General Public License, version 2
# or later which is available at
# https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

# @file    runner.py
# @author  Jakob Erdmann
# @date    2018-09-27

from __future__ import print_function
from __future__ import absolute_import
import os
import sys
from pprint import pprint

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci  # noqa
import sumolib  # noqa
import traci.constants as tc  # noqa

sumoBinary = sumolib.checkBinary('sumo')
traci.start([sumoBinary,
             "-n", "input_net.net.xml",
             "-r", "input_routes.rou.xml",
             "--default.speeddev", "0",
             "--no-step-log",
             ])

vehID = "ego"
traci.vehicle.subscribeContext(vehID, tc.CMD_GET_VEHICLE_VARIABLE, 50,
                               [tc.VAR_SPEED, tc.VAR_LANE_ID, tc.VAR_POSITION])

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    pprint(traci.vehicle.getContextSubscriptionResults(vehID))

traci.close()
