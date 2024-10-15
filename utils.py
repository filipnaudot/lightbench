class Colors:
    # BACKGROUND
    BACKGROUND_LIGHT_BLUE = '\033[104m'
    BACKGROUND_GRAY       = '\033[100m'
    # TEXT
    WHITE_TEXT            = '\033[97m'
    GREEN_TEXT            = '\033[92m'
    CYAN_TEXT             = '\033[96m'
    YELLOW_TEXT           = '\033[93m'
    RED_TEXT              = '\033[91m'

    RESET                 = '\033[0m'

class Printer:
    def print_green(input, end="\n"):
        print(f"{Colors.GREEN_TEXT}{input}{Colors.RESET}", end=end)

    def print_yellow(input, end="\n"):
        print(f"{Colors.YELLOW_TEXT}{input}{Colors.RESET}", end=end)

    def print_red(input, end="\n"):
        print(f"{Colors.RED_TEXT}{input}{Colors.RESET}", end=end)

    def print_cyan(input, end="\n"):
        print(f"{Colors.CYAN_TEXT}{input}{Colors.RESET}", end=end)