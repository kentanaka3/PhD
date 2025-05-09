from constants import *
# LEVEL : Level of the error
#   WARNING / FATAL
# pre : Who, What, Where, When, Why or How
#   What:
#     ASSIGN / UNABLE / NOTABLE / UNKNOWN
# type : ID
# post : Random information
ERR_MSG = "{LEVEL} v1: [{pre}] {type} {post}"

ERRORS = {
    FATAL_STR: {

    },
    WARNING_STR: {
        ASSIGN_STR: ERR_MSG.format(
            LEVEL=WARNING_STR, pre="{pre}" + ASSIGN_STR,
            type="value ({value}) to ({key})", post="{post}"),
        NOTABLE_STR: ERR_MSG.format(LEVEL=WARNING_STR, pre="{pre}" + NOTABLE_STR,
                                    type="value ({value}) in ({key})",
                                    post="{post}"),
        UNABLE_STR: ERR_MSG.format(LEVEL=WARNING_STR, pre="{pre}" + UNABLE_STR,
                                   type="to {verb} {type}", post="{post}"),
        UNKNOWN_STR: ERR_MSG.format(LEVEL=WARNING_STR, pre="{pre}" + UNKNOWN_STR,
                                    type="value ({value}) in ({key})",
                                    post="{post}"),
    }
}
