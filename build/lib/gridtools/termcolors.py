from itertools import cycle


class TermColor:
    """
    Class providing constants and functions containing ANSI color escape codes
    to increase saliency of important shell messages.
    """

    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"

    @classmethod
    def err(cls, string):
        """Print a bold red terminal message.
        Returned string must be inside a print statement for formatting to work.

        Args:
              string (_string_): A quoted string containing the message.

        Returns:
              _string_: A string with added ANSI escape characters for text formatting.
        """
        return f"{cls.RED}{cls.BOLD}{string}{cls.END}"

    @classmethod
    def warn(cls, string):
        """Print a bold yellow terminal message.
        Returned string must be inside a print statement for formatting to work.

        Args:
              string (_string_): A quoted string containing the message.

        Returns:
              _string_: A string with added ANSI escape characters for text formatting.
        """
        return f"{cls.YELLOW}{cls.BOLD}{string}{cls.END}"

    @classmethod
    def succ(cls, string):
        """Print a bold green terminal message.
        Returned string must be inside a print statement for formatting to work.

        Args:
              string (_string_): A quoted string containing the message.

        Returns:
              _string_: A string with added ANSI escape characters for text formatting.
        """
        return f"{cls.GREEN}{cls.BOLD}{string}{cls.END}"

    @classmethod
    def rainb(cls, string):
        """Print works seperated by spaces in rainbow colors.
        Returned string must be inside a print statement for formatting to work.

        Args:
              string (_string_): A quoted string containing the message.

        Returns:
              _string_: A string with added ANSI escape characters for text formatting.
        """
        colors = [
            cls.BLUE,
            cls.DARKCYAN,
            cls.CYAN,
            cls.GREEN,
            cls.YELLOW,
            cls.RED,
            cls.PURPLE,
        ]

        spltstr = string.split()
        word_count = len(spltstr)
        newstr = ""

        for count, color in enumerate(cycle(colors)):
            # print(count, color)
            if count + 1 > word_count:
                newstr += cls.END
                break
            newstr += color + spltstr[count] + " "
        return newstr
