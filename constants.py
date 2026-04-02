
ACTIONS = [0, 1, 2]  # turn left, straight, turn right
ACTION_STRING = ["L", "S", "R"]

DIRECTIONS = [0, 1, 2, 3]  # up, right, down, left

SNAKE_FEATURE_COLS = [

    #numbers for norm features represent distance in direction divided by max possible distance.
    "food_front/back_norm", # positive is food infront, negative is back
    "food_right/left_norm", # positive is food right, negative is left
    "obstacle_front",
    "obstacle_right",
    "obstacle_left",
]