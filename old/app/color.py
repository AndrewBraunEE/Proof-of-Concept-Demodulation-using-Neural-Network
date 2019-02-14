from enum import Enum

FOREGROUND_OFFSET = 30
BACKGROUND_OFFSET = 40

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

class Colors(Enum):
    BLACK = 0
    RED = 1
    GREEN = 2
    YELLOW = 3
    BLUE = 4
    MAGENTA = 5
    CYAN = 6
    WHITE = 7

# TODO Implement background color (what corresponds to no background)?
def color_text(text, fg, bg=None):

    fg = fg.value + FOREGROUND_OFFSET
    #bg = bg.value + BACKGROUND_OFFSET

    colored = (COLOR_SEQ % fg) + text + RESET_SEQ
    return colored

def red(text):
    return color_text(text, Colors.RED)

def green(text):
    return color_text(text, Colors.GREEN)

def blue(text):
    return color_text(text, Colors.BLUE)

def yellow(text):
    return color_text(text, Colors.YELLOW)
