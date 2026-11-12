SINGLE_LINE_WIDTH = 8.23
DOUBLE_LINE_WIDTH = 10.97
HALF_COURT_LINE_HEIGHT = 11.88
SERVICE_LINE_WIDTH = 6.4
DOUBLE_ALLY_DIFFERENCE = 1.37
NO_MANS_LAND_HEIGHT = 5.48
COURT_NET_HEIGHT = 0.914
DISTANCE_BETWEEN_POINT_12_AND_13 = 12.8

PLAYER_1_HEIGHT_METERS = 1.88
PLAYER_2_HEIGHT_METERS = 1.91



# ---- homography / net / in-out ----
COURT_LENGTH = HALF_COURT_LINE_HEIGHT * 2                  # 23.76 m baseline to baseline
NET_POST_OUTSIDE_DOUBLES = 0.914                           # posts stand 0.914 m outside the doubles lines
NET_HEIGHT_AT_POST = 1.07                                  # net is 0.914 m in the middle (COURT_NET_HEIGHT)

# singles court = safe landing area (the lines count as IN)
SINGLES_LEFT = DOUBLE_ALLY_DIFFERENCE                      # 1.37 m
SINGLES_RIGHT = DOUBLE_LINE_WIDTH - DOUBLE_ALLY_DIFFERENCE # 9.60 m
LINE_TOLERANCE_METERS = 0.033                              # ball radius: touching the line is IN

MY_SIDE = 'near'                                           # 'near' = camera side (bottom), 'far' = top