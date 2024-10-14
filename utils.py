class Colors:
    # BACKGROUND
    BACKGROUND_LIGHT_BLUE = '\033[104m'
    BACKGROUND_GRAY       = '\033[100m'
    # TEXT
    WHITE_TEXT            = '\033[97m'
    GREEN_TEXT            = '\033[92m'
    RED_TEXT              = '\033[91m'

    RESET                 = '\033[0m'


def print_green(input, end="\n"):
    print(f"{Colors.GREEN_TEXT}{input}{Colors.RESET}", end=end)

def print_red(input, end="\n"):
    print(f"{Colors.RED_TEXT}{input}{Colors.RESET}", end=end)