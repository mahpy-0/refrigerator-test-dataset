import json
import random
import itertools
from pathlib import Path

import pandas as pd


# ============================================================
# تنظیمات
# ============================================================

SEED = 42
random.seed(SEED)

# برای تست اولیه بهتر است 3 یا 4 باشد.
# برای تمام ترکیب‌های ممکن 17 قرار دهید.
MAX_COMBINATION_SIZE = 4

GENERATE_INVALID_SET_CASES = True
GENERATE_READ_ONLY_GETS = True

OUTPUT_CSV = "./data/gpt-5.5-instant/refrigerator_rag_test_dataset_fa.csv"
# OUTPUT_JSONL = "refrigerator_rag_test_dataset_fa.jsonl"


# ============================================================
# فیلدهای JSON
# ============================================================

ALL_FIELDS = [
    "message",
    "refTemp",
    "frzTemp",
    "iceType",
    "smartMode",
    "water",
    "supperFrz",
    "supperRef",
    "iceMaker",
    "light",
    "childLock",
    "magicZone",
    "filterReset",
    "sleepMode",
    "Hygiene",
    "doorAlarm",
    "extraRef",
    "vacation",
    "iceFull",
    "filterTime",
    "environmentTemp",
    "ragImageURL",
]


# 17 پارامتر قابل تنظیم
WRITABLE_PARAMETERS = [
    "refTemp",
    "frzTemp",
    "iceType",
    "smartMode",
    "water",
    "supperFrz",
    "supperRef",
    "iceMaker",
    "light",
    "childLock",
    "magicZone",
    "filterReset",
    "sleepMode",
    "Hygiene",
    "doorAlarm",
    "extraRef",
    "vacation",
]


# پارامترهای فقط خواندنی
READ_ONLY_PARAMETERS = [
    "iceFull",
    "filterTime",
    "environmentTemp",
]


# ============================================================
# اطلاعات پارامترها
# ============================================================

PARAMS = {
    "refTemp": {
        "display": "دمای یخچال",
        "kind": "number",
        "valid_values": [1, 2, 3, 4, 5, 6, 7],
        "invalid_values": [-10, -1, 0, 8, 9, 15, 100],
        "unit": "درجه سانتی‌گراد",

        "set_templates": [
            "دمای یخچال را روی {value} درجه تنظیم کن",
            "دمای یخچال رو بذار روی {value}",
            "یخچال رو روی {value} درجه تنظیم کن",
            "دمای قسمت یخچال را {value} درجه کن",
            "می‌خوام دمای یخچال {value} درجه باشه",
        ],

        "get_templates": [
            "دمای یخچال چنده؟",
            "دمای فعلی یخچال رو بهم بگو",
            "یخچال روی چند درجه تنظیم شده؟",
            "الان دمای قسمت یخچال چقدره؟",
            "یخچال چند درجه است؟",
        ],
    },

    "frzTemp": {
        "display": "دمای فریزر",
        "kind": "number",
        "valid_values": [-24, -23, -22, -21, -20, -19, -18, -17, -16],
        "invalid_values": [-40, -30, -25, -15, -10, 0, 10],
        "unit": "درجه سانتی‌گراد",

        "set_templates": [
            "دمای فریزر را روی {value} درجه تنظیم کن",
            "دمای فریزر رو بذار روی {value}",
            "فریزر رو روی {value} درجه تنظیم کن",
            "دمای قسمت فریزر را {value} درجه کن",
            "می‌خوام فریزر روی {value} درجه باشه",
        ],

        "get_templates": [
            "دمای فریزر چنده؟",
            "دمای فعلی فریزر رو بهم بگو",
            "فریزر روی چند درجه تنظیم شده؟",
            "الان فریزر چند درجه است؟",
            "دمای قسمت فریزر چقدره؟",
        ],
    },

    "iceType": {
        "display": "نوع یخ",
        "kind": "enum",
        "valid_values": ["Crushed", "Cubed"],
        "invalid_values": [
            "Round",
            "Flaked",
            "Snow",
            "Liquid",
            "Large",
        ],
        "unit": None,

        # value_fa استفاده خواهد شد.
        "set_templates": [
            "نوع یخ رو روی {value_fa} بذار",
            "یخ {value_fa} می‌خوام",
            "خروجی یخ رو {value_fa} کن",
            "دستگاه رو برای یخ {value_fa} تنظیم کن",
        ],

        "get_templates": [
            "نوع یخ خروجی چیه؟",
            "الان یخ خرد شده میده یا قالبی؟",
            "نوع یخ دستگاه رو بهم بگو",
            "چه نوع یخی از دستگاه خارج میشه؟",
        ],
    },

    "smartMode": {
        "display": "حالت هوشمند",
        "kind": "enum",
        "valid_values": [
            "Default",
            "Power Saving",
            "Guest",
            "Night",
            "Novelty",
            "Off",
        ],
        "invalid_values": [
            "Turbo",
            "Gaming",
            "Maximum",
            "Weekend",
            "Automatic Plus",
        ],
        "unit": None,

        "set_templates": [
            "حالت هوشمند رو روی {value_fa} بذار",
            "حالت {value_fa} رو فعال کن",
            "یخچال رو روی حالت {value_fa} تنظیم کن",
            "از حالت {value_fa} استفاده کن",
        ],

        "get_templates": [
            "الان حالت هوشمند روی چیه؟",
            "یخچال الان در چه حالتی کار می‌کنه؟",
            "حالت فعلی دستگاه چیه؟",
            "اسمارت مود الان چیه؟",
        ],
    },
}


# ============================================================
# نگاشت مقادیر API به فارسی
# ============================================================

ENUM_FA = {
    "iceType": {
        "Crushed": "خرد شده",
        "Cubed": "قالبی",
    },

    "smartMode": {
        "Default": "پیش‌فرض",
        "Power Saving": "صرفه‌جویی در مصرف انرژی",
        "Guest": "مهمان",
        "Night": "شب",
        "Novelty": "هوشمند شخصی‌سازی‌شده",
        "Off": "خاموش",
    },
}


# ============================================================
# اطلاعات پارامترهای boolean
# ============================================================

BOOLEAN_INFO = {
    "water": {
        "display": "آب‌سردکن",
        "aliases": [
            "آب‌سردکن",
            "آبریز",
            "خروجی آب",
        ],
    },

    "supperFrz": {
        "display": "انجماد سریع",
        "aliases": [
            "انجماد سریع",
            "فریز سریع",
            "سوپر فریز",
        ],
    },

    "supperRef": {
        "display": "سرمایش سریع یخچال",
        "aliases": [
            "سرمایش سریع یخچال",
            "خنک‌کنندگی سریع",
            "سوپر کول",
        ],
    },

    "iceMaker": {
        "display": "یخ‌ساز",
        "aliases": [
            "یخ‌ساز",
            "سیستم یخ‌ساز",
        ],
    },

    "light": {
        "display": "چراغ آب‌سردکن",
        "aliases": [
            "چراغ آب‌سردکن",
            "نور آبریز",
            "چراغ روی در",
        ],
    },

    "childLock": {
        "display": "قفل کودک",
        "aliases": [
            "قفل کودک",
            "قفل ایمنی کودک",
        ],
    },

    "magicZone": {
        "display": "محفظه صفر درجه",
        "aliases": [
            "محفظه صفر درجه",
            "مجیک زون",
            "چیلر",
            "محفظه تازگی",
        ],
    },

    "filterReset": {
        "display": "ریست فیلتر آب",
        "aliases": [
            "ریست فیلتر",
            "بازنشانی فیلتر آب",
            "ریست شمارنده فیلتر",
        ],
    },

    "sleepMode": {
        "display": "حالت خواب",
        "aliases": [
            "حالت خواب",
            "حالت کم‌صدا",
            "حالت شبانه کم‌صدا",
        ],
    },

    "Hygiene": {
        "display": "تصفیه هوا",
        "aliases": [
            "تصفیه هوا",
            "فیلتر هوا",
            "هایژن",
            "پلاسما",
            "تصفیه‌کننده هوا",
        ],
    },

    "doorAlarm": {
        "display": "هشدار باز ماندن در",
        "aliases": [
            "هشدار در",
            "آلارم در",
            "هشدار باز ماندن در",
        ],
    },

    "extraRef": {
        "display": "سرمایش اضافه یخچال",
        "aliases": [
            "سرمایش اضافه",
            "خنک‌کنندگی اضافه یخچال",
            "اکسترا کولینگ",
        ],
    },

    "vacation": {
        "display": "حالت مسافرت",
        "aliases": [
            "حالت مسافرت",
            "حالت تعطیلات",
            "حالت اکو",
        ],
    },
}


# ============================================================
# ساخت metadata پارامترهای boolean
# ============================================================

for param, info in BOOLEAN_INFO.items():

    PARAMS[param] = {
        "display": info["display"],
        "kind": "bool",
        "valid_values": [True, False],

        # این مقادیر مستقیماً وارد template نمی‌شوند.
        "invalid_values": [
            "نامشخص",
            "نصفه",
            "گاهی",
        ],

        "unit": None,

        "get_templates": [
            "آیا {alias} روشنه؟",
            "{alias} فعاله؟",
            "وضعیت {alias} چیه؟",
            "الان {alias} روشنه یا خاموش؟",
            "بهم بگو {alias} فعاله یا نه",
        ],
    }


# ============================================================
# Read-only metadata
# ============================================================

PARAMS["iceFull"] = {
    "display": "وضعیت پر بودن مخزن یخ",
    "kind": "bool",
    "valid_values": [True, False],
    "invalid_values": [],
    "unit": None,

    "get_templates": [
        "مخزن یخ پره؟",
        "سبد یخ پر شده؟",
        "آیا مخزن یخ پر است؟",
        "وضعیت پر بودن مخزن یخ رو بهم بگو",
    ],
}


PARAMS["filterTime"] = {
    "display": "زمان کارکرد فیلتر آب",
    "kind": "integer",
    "valid_values": [
        0,
        1,
        24,
        100,
        500,
        1000,
        3000,
    ],
    "invalid_values": [],
    "unit": "ساعت",

    "get_templates": [
        "فیلتر آب چند ساعت کار کرده؟",
        "زمان کارکرد فیلتر آب چقدره؟",
        "چند ساعت از عمر فیلتر آب گذشته؟",
        "ساعت کارکرد فیلتر رو بهم بگو",
    ],
}


PARAMS["environmentTemp"] = {
    "display": "دمای محیط",
    "kind": "number",
    "valid_values": [
        -5,
        0,
        10,
        20,
        25,
        30,
        40,
        50,
    ],
    "invalid_values": [],
    "unit": "درجه سانتی‌گراد",

    "get_templates": [
        "دمای محیط اطراف یخچال چنده؟",
        "دمای اتاق کنار یخچال چقدره؟",
        "دمای محیط رو بهم بگو",
        "سنسور دمای محیط چند درجه نشون میده؟",
    ],
}


# ============================================================
# توابع کمکی
# ============================================================

def empty_payload():
    """
    تمام پارامترها را null قرار می‌دهد.
    """
    return {
        field: None
        for field in ALL_FIELDS
    }


def json_to_string(data):
    """
    تبدیل dictionary به JSON string با پشتیبانی صحیح از فارسی.
    """
    return json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def random_current_value(param):

    if param == "filterTime":
        return random.randint(0, 5000)

    if param == "environmentTemp":
        return random.randint(-5, 50)

    return random.choice(
        PARAMS[param]["valid_values"]
    )


def different_valid_value(param, current_value):

    candidates = [
        value
        for value in PARAMS[param]["valid_values"]
        if value != current_value
    ]

    if candidates:
        return random.choice(candidates)

    return random.choice(
        PARAMS[param]["valid_values"]
    )


def value_to_farsi(param, value):

    if param in ENUM_FA:
        return ENUM_FA[param].get(
            value,
            str(value)
        )

    if isinstance(value, bool):
        return "روشن" if value else "خاموش"

    return str(value)


def human_value(param, value):

    if isinstance(value, bool):
        return "روشن" if value else "خاموش"

    value_text = value_to_farsi(
        param,
        value,
    )

    unit = PARAMS[param].get("unit")

    if unit:
        return f"{value_text} {unit}"

    return value_text


# ============================================================
# ساخت سوال SET
# ============================================================

def make_single_set_phrase(param, value):

    meta = PARAMS[param]

    # --------------------------------------------
    # Boolean
    # --------------------------------------------
    if meta["kind"] == "bool":

        alias = random.choice(
            BOOLEAN_INFO[param]["aliases"]
        )

        # valid boolean
        if isinstance(value, bool):

            if value:
                templates = [
                    "{alias} رو روشن کن",
                    "{alias} رو فعال کن",
                    "{alias} را روشن کن",
                    "لطفاً {alias} رو فعال کن",
                    "می‌خوام {alias} روشن باشه",
                ]

            else:
                templates = [
                    "{alias} رو خاموش کن",
                    "{alias} رو غیرفعال کن",
                    "{alias} را خاموش کن",
                    "لطفاً {alias} رو غیرفعال کن",
                    "می‌خوام {alias} خاموش باشه",
                ]

            template = random.choice(
                templates
            )

            return template.format(
                alias=alias
            )

        # invalid / semantically impossible boolean request
        invalid_templates = [
            "{alias} رو نصفه روشن کن",
            "{alias} رو روی حالت شاید بذار",
            "{alias} رو ۵ قرار بده",
            "{alias} رو همزمان روشن و خاموش کن",
        ]

        return random.choice(
            invalid_templates
        ).format(
            alias=alias
        )

    # --------------------------------------------
    # Enum / Number
    # --------------------------------------------

    template = random.choice(
        meta["set_templates"]
    )

    return template.format(
        value=value,
        value_fa=value_to_farsi(
            param,
            value,
        ),
    )


# ============================================================
# ساخت سوال GET
# ============================================================

def make_single_get_phrase(param):

    meta = PARAMS[param]

    template = random.choice(
        meta["get_templates"]
    )

    if "{alias}" in template:

        alias = random.choice(
            BOOLEAN_INFO[param]["aliases"]
        )

        return template.format(
            alias=alias
        )

    return template


# ============================================================
# اتصال چند درخواست
# ============================================================

def join_requests(parts):

    if len(parts) == 1:
        return parts[0]

    if len(parts) == 2:
        return (
            parts[0].rstrip("؟?")
            + " و "
            + parts[1]
        )

    return "، ".join(parts[:-1]) + " و " + parts[-1]


# ============================================================
# DeviceState
# ============================================================

def make_device_state(
    message,
    relevant_values=None
):

    data = empty_payload()

    data["message"] = message

    if relevant_values:

        for param, value in relevant_values.items():
            data[param] = value

    # طبق requirement همیشه null
    data["ragImageURL"] = None

    return data


# ============================================================
# iot_middleware_data
# ============================================================

def make_middleware(
    message,
    changes=None
):

    data = empty_payload()

    data["message"] = message

    if changes:

        for param, value in changes.items():
            data[param] = value

    data["ragImageURL"] = None

    return data


# ============================================================
# پاسخ GET
# ============================================================

def make_get_response(values):

    parts = []

    for param, value in values.items():

        display = PARAMS[param]["display"]

        if isinstance(value, bool):

            status = (
                "روشن"
                if value
                else "خاموش"
            )

            parts.append(
                f"{display} {status} است"
            )

        else:

            parts.append(
                f"{display} {human_value(param, value)} است"
            )

    if len(parts) == 1:
        return parts[0] + "."

    return "، ".join(parts) + "."


# ============================================================
# پاسخ SET موفق
# ============================================================

def make_set_success_response(changes):

    if len(changes) == 1:

        param, value = next(
            iter(changes.items())
        )

        display = PARAMS[param]["display"]

        if isinstance(value, bool):

            state = (
                "فعال شد"
                if value
                else "غیرفعال شد"
            )

            return (
                f"{display} با موفقیت "
                f"{state}."
            )

        return (
            f"{display} با موفقیت روی "
            f"{human_value(param, value)} "
            f"تنظیم شد."
        )

    parts = []

    for param, value in changes.items():

        display = PARAMS[param]["display"]

        if isinstance(value, bool):

            text = (
                "روشن"
                if value
                else "خاموش"
            )

        else:

            text = human_value(
                param,
                value,
            )

        parts.append(
            f"{display}: {text}"
        )

    return (
        "تنظیمات با موفقیت اعمال شد: "
        + "، ".join(parts)
        + "."
    )


# ============================================================
# پاسخ SET نامعتبر
# ============================================================

def make_invalid_set_response(
    invalid_items
):

    reasons = []

    for param, value in invalid_items.items():

        display = PARAMS[param]["display"]

        if param == "refTemp":

            reasons.append(
                f"{display} نمی‌تواند روی "
                f"{value} تنظیم شود؛ محدوده مجاز "
                f"از ۱ تا ۷ درجه است"
            )

        elif param == "frzTemp":

            reasons.append(
                f"{display} نمی‌تواند روی "
                f"{value} تنظیم شود؛ محدوده مجاز "
                f"از ۲۴- تا ۱۶- درجه است"
            )

        elif param == "iceType":

            reasons.append(
                "نوع یخ درخواستی معتبر نیست؛ "
                "نوع یخ باید خرد شده یا قالبی باشد"
            )

        elif param == "smartMode":

            reasons.append(
                "حالت هوشمند درخواستی معتبر نیست"
            )

        elif PARAMS[param]["kind"] == "bool":

            reasons.append(
                f"برای {display} باید مشخص شود "
                f"که روشن یا خاموش شود"
            )

        else:

            reasons.append(
                f"مقدار درخواست‌شده برای "
                f"{display} معتبر نیست"
            )

    return (
        "تنظیمات اعمال نشد. "
        + "؛ ".join(reasons)
        + "."
    )


# ============================================================
# ذخیره row
# ============================================================

rows = []


def add_row(
    device_state,
    scenario_type,
    middleware
):

    rows.append({
        "index": len(rows),
        "DeviceState": json_to_string(
            device_state
        ),
        "type": scenario_type,
        "iot_middleware_data": json_to_string(
            middleware
        ),
    })


# ============================================================
# GET
# ============================================================

def generate_get_case(parameters):

    current_values = {
        param: random_current_value(param)
        for param in parameters
    }

    questions = [
        make_single_get_phrase(param)
        for param in parameters
    ]

    message = join_requests(
        questions
    )

    device_state = make_device_state(
        message,
        current_values,
    )

    response = make_get_response(
        current_values
    )

    middleware = make_middleware(
        response,
        None,
    )

    add_row(
        device_state,
        "get",
        middleware,
    )


# ============================================================
# SET صحیح
# ============================================================

def generate_valid_set_case(parameters):

    current_values = {
        param: random_current_value(param)
        for param in parameters
    }

    requested_values = {
        param: different_valid_value(
            param,
            current_values[param]
        )
        for param in parameters
    }

    requests = [
        make_single_set_phrase(
            param,
            requested_values[param]
        )
        for param in parameters
    ]

    message = join_requests(
        requests
    )

    device_state = make_device_state(
        message,
        current_values,
    )

    response = make_set_success_response(
        requested_values
    )

    middleware = make_middleware(
        response,
        requested_values,
    )

    add_row(
        device_state,
        "set_valid",
        middleware,
    )


# ============================================================
# SET نامعتبر
# ============================================================

def generate_invalid_set_case(parameters):

    candidates = [
        param
        for param in parameters
        if PARAMS[param]["invalid_values"]
    ]

    if not candidates:
        return

    # حداقل یک پارامتر خراب می‌شود
    invalid_param = random.choice(
        candidates
    )

    current_values = {
        param: random_current_value(param)
        for param in parameters
    }

    requested_values = {}

    for param in parameters:

        if param == invalid_param:

            requested_values[param] = random.choice(
                PARAMS[param]["invalid_values"]
            )

        else:

            requested_values[param] = (
                different_valid_value(
                    param,
                    current_values[param]
                )
            )

    requests = [
        make_single_set_phrase(
            param,
            requested_values[param]
        )
        for param in parameters
    ]

    message = join_requests(
        requests
    )

    device_state = make_device_state(
        message,
        current_values,
    )

    response = make_invalid_set_response({
        invalid_param:
            requested_values[invalid_param]
    })

    # مهم:
    # رفتار atomic
    # هیچ تغییری اعمال نمی‌شود.
    middleware = make_middleware(
        response,
        None,
    )

    add_row(
        device_state,
        "set_invalid",
        middleware,
    )


# ============================================================
# تست boundary دما
# ============================================================

def generate_temperature_boundary_cases():

    tests = {
        "refTemp": {
            "valid": [
                1,
                7,
            ],
            "invalid": [
                0,
                8,
                -1,
                100,
            ],
        },

        "frzTemp": {
            "valid": [
                -24,
                -16,
            ],
            "invalid": [
                -25,
                -15,
                0,
                -100,
            ],
        },
    }

    for param, values in tests.items():

        # valid
        for requested in values["valid"]:

            current = random_current_value(
                param
            )

            message = (
                make_single_set_phrase(
                    param,
                    requested
                )
            )

            device_state = (
                make_device_state(
                    message,
                    {param: current},
                )
            )

            middleware = (
                make_middleware(
                    make_set_success_response({
                        param: requested
                    }),
                    {param: requested},
                )
            )

            add_row(
                device_state,
                "set_valid_boundary",
                middleware,
            )

        # invalid
        for requested in values["invalid"]:

            current = random_current_value(
                param
            )

            message = (
                make_single_set_phrase(
                    param,
                    requested
                )
            )

            device_state = (
                make_device_state(
                    message,
                    {param: current},
                )
            )

            middleware = (
                make_middleware(
                    make_invalid_set_response({
                        param: requested
                    }),
                    None,
                )
            )

            add_row(
                device_state,
                "set_invalid_boundary",
                middleware,
            )


# ============================================================
# تلاش برای SET پارامتر read-only
# ============================================================

def generate_read_only_set_attempts():

    cases = [
        (
            "iceFull",
            "مخزن یخ رو روی پر تنظیم کن",
        ),
        (
            "iceFull",
            "وضعیت مخزن یخ رو خالی کن",
        ),
        (
            "filterTime",
            "زمان کارکرد فیلتر رو صفر کن",
        ),
        (
            "filterTime",
            "ساعت فیلتر آب رو روی ۱۰۰ بذار",
        ),
        (
            "environmentTemp",
            "دمای محیط رو روی ۲۰ درجه تنظیم کن",
        ),
    ]

    for param, message in cases:

        current = random_current_value(
            param
        )

        device_state = make_device_state(
            message,
            {param: current},
        )

        response = (
            f"{PARAMS[param]['display']} "
            f"فقط خواندنی است و امکان تغییر آن "
            f"وجود ندارد. هیچ تنظیمی اعمال نشد."
        )

        middleware = make_middleware(
            response,
            None,
        )

        add_row(
            device_state,
            "set_read_only_rejected",
            middleware,
        )


# ============================================================
# سوال‌های مبهم
# ============================================================

VAGUE_CASES = [
    (
        "روشنش کن",
        "لطفاً مشخص کنید کدام قابلیت را می‌خواهید روشن کنم."
    ),
    (
        "خاموشش کن",
        "لطفاً مشخص کنید کدام قابلیت را می‌خواهید خاموش کنم."
    ),
    (
        "سردترش کن",
        "منظورتان یخچال است یا فریزر؟ لطفاً دمای موردنظر را هم مشخص کنید."
    ),
    (
        "گرم‌ترش کن",
        "لطفاً مشخص کنید دمای یخچال یا فریزر را می‌خواهید تغییر دهید و دمای موردنظر چقدر است."
    ),
    (
        "دماش رو عوض کن",
        "لطفاً مشخص کنید دمای یخچال یا فریزر را می‌خواهید تغییر دهید و مقدار موردنظر چقدر است."
    ),
    (
        "حالتش رو عوض کن",
        "لطفاً مشخص کنید کدام حالت یا تنظیم دستگاه را می‌خواهید تغییر دهید."
    ),
    (
        "بهترش کن",
        "لطفاً مشخص کنید کدام تنظیم یخچال را می‌خواهید تغییر دهید."
    ),
    (
        "یه کاری کن مصرفش کمتر شه",
        "آیا می‌خواهید حالت صرفه‌جویی در مصرف انرژی را فعال کنم؟"
    ),
    (
        "تنظیمش کن",
        "لطفاً مشخص کنید کدام تنظیم دستگاه و چه مقداری مدنظر شماست."
    ),
    (
        "تغییرش بده",
        "لطفاً مشخص کنید کدام تنظیم را می‌خواهید تغییر دهید."
    ),
    (
        "همونو فعال کن",
        "لطفاً مشخص کنید کدام قابلیت را می‌خواهید فعال کنم."
    ),
    (
        "اون یکی رو خاموش کن",
        "لطفاً مشخص کنید کدام قابلیت را می‌خواهید خاموش کنم."
    ),
    (
        "یخ رو عوض کن",
        "لطفاً مشخص کنید یخ خرد شده می‌خواهید یا یخ قالبی."
    ),
    (
        "بذار روی حالت خوب",
        "لطفاً مشخص کنید کدام حالت هوشمند مدنظر شماست."
    ),
    (
        "دمای مناسب بذار",
        "لطفاً مشخص کنید دمای یخچال یا فریزر را می‌خواهید تنظیم کنید و مقدار موردنظر چقدر است."
    ),
]


def generate_vague_cases():

    for message, response in VAGUE_CASES:

        device_state = make_device_state(
            message,
            None,
        )

        middleware = make_middleware(
            response,
            None,
        )

        add_row(
            device_state,
            "vague",
            middleware,
        )


# ============================================================
# سوال‌های خارج از حوزه
# ============================================================

IRRELEVANT_MESSAGES = [
    "امروز هوا چطوره؟",
    "یه جوک برام تعریف کن",
    "پایتخت ژاپن کجاست؟",
    "دو به علاوه دو چند میشه؟",
    "قیمت دلار چنده؟",
    "برام یه برنامه پایتون بنویس",
    "یه فیلم خوب معرفی کن",
    "موزیک پخش کن",
    "به دوستم پیام بده",
    "نتیجه فوتبال چی شد؟",
    "رئیس جمهور فرانسه کیه؟",
    "یه بلیط هواپیما برام رزرو کن",
]


def generate_irrelevant_cases():

    for message in IRRELEVANT_MESSAGES:

        device_state = make_device_state(
            message,
            None,
        )

        response = (
            "من می‌توانم درباره تنظیمات و وضعیت "
            "یخچال هوشمند کمک کنم. لطفاً درخواست "
            "مرتبط با یخچال مطرح کنید."
        )

        middleware = make_middleware(
            response,
            None,
        )

        add_row(
            device_state,
            "irrelevant",
            middleware,
        )


# ============================================================
# Alias GET
# ============================================================

ALIAS_GET_CASES = [
    (
        "supperRef",
        "سوپر کول روشنه؟",
    ),
    (
        "supperRef",
        "خنک‌کنندگی سریع فعاله؟",
    ),
    (
        "supperFrz",
        "سوپر فریز فعاله؟",
    ),
    (
        "supperFrz",
        "فریز سریع روشنه؟",
    ),
    (
        "magicZone",
        "چیلر روشنه؟",
    ),
    (
        "magicZone",
        "محفظه تازگی فعاله؟",
    ),
    (
        "light",
        "چراغ آبریز روشنه؟",
    ),
    (
        "childLock",
        "قفل ایمنی کودک روشنه؟",
    ),
    (
        "sleepMode",
        "حالت کم‌صدا فعاله؟",
    ),
    (
        "vacation",
        "حالت تعطیلات روشنه؟",
    ),
    (
        "vacation",
        "اکو مود فعاله؟",
    ),
    (
        "Hygiene",
        "تصفیه‌کننده هوا روشنه؟",
    ),
    (
        "Hygiene",
        "پلاسما فعاله؟",
    ),
    (
        "doorAlarm",
        "هشدار باز موندن در فعاله؟",
    ),
    (
        "iceType",
        "الان یخ پودری میده یا قالبی؟",
    ),
]


def generate_alias_get_cases():

    for param, message in ALIAS_GET_CASES:

        current = random_current_value(
            param
        )

        device_state = make_device_state(
            message,
            {param: current},
        )

        response = make_get_response({
            param: current
        })

        middleware = make_middleware(
            response,
            None,
        )

        add_row(
            device_state,
            "get_alias",
            middleware,
        )


# ============================================================
# درخواست‌های semantic برای smartMode
# ============================================================

SMART_MODE_CASES = [
    (
        "می‌خوام مصرف برق یخچال کمتر بشه",
        "Power Saving",
    ),
    (
        "یه حالت کم مصرف بذار",
        "Power Saving",
    ),
    (
        "مهمون داریم و در یخچال خیلی باز و بسته میشه",
        "Guest",
    ),
    (
        "تازه خرید کردم و یخچال رو پر کردم",
        "Guest",
    ),
    (
        "شب‌ها صدای یخچال اذیتم می‌کنه، کم‌صداتر کار کنه",
        "Night",
    ),
    (
        "شب یخچال آروم‌تر کار کنه",
        "Night",
    ),
    (
        "یخچال رو برگردون به حالت عادی کارخانه",
        "Default",
    ),
    (
        "حالت معمولی و متعادل رو فعال کن",
        "Default",
    ),
    (
        "حالت هوشمندی که عادت‌های منو یاد می‌گیره فعال کن",
        "Novelty",
    ),
    (
        "یه حالت مدرن و شخصی‌سازی‌شده برای استفاده من بذار",
        "Novelty",
    ),
    (
        "حالت‌های هوشمند رو خاموش کن و تنظیمات دستی باشه",
        "Off",
    ),
]


def generate_smart_mode_semantic_cases():

    for message, requested in SMART_MODE_CASES:

        current = random_current_value(
            "smartMode"
        )

        device_state = make_device_state(
            message,
            {"smartMode": current},
        )

        response = make_set_success_response({
            "smartMode": requested
        })

        middleware = make_middleware(
            response,
            {"smartMode": requested},
        )

        add_row(
            device_state,
            "set_semantic",
            middleware,
        )


# ============================================================
# GET ترکیبی شامل read-only
# ============================================================

def generate_mixed_readonly_get_cases():

    cases = [
        (
            "refTemp",
            "environmentTemp",
        ),
        (
            "frzTemp",
            "filterTime",
        ),
        (
            "iceMaker",
            "iceFull",
        ),
        (
            "refTemp",
            "frzTemp",
            "environmentTemp",
        ),
        (
            "iceMaker",
            "iceFull",
            "filterTime",
        ),
    ]

    for params in cases:
        generate_get_case(params)


# ============================================================
# ساخت تمام combination ها
# ============================================================

def generate_parameter_combinations():

    max_size = min(
        MAX_COMBINATION_SIZE,
        len(WRITABLE_PARAMETERS)
    )

    for size in range(
        1,
        max_size + 1
    ):

        print(
            f"Generating {size}-parameter combinations..."
        )

        for combo in itertools.combinations(
            WRITABLE_PARAMETERS,
            size
        ):

            # GET
            generate_get_case(
                combo
            )

            # SET valid
            generate_valid_set_case(
                combo
            )

            # SET invalid
            if GENERATE_INVALID_SET_CASES:

                generate_invalid_set_case(
                    combo
                )


# ============================================================
# Dataset validation
# ============================================================

def validate_dataset(df):

    print("\nValidating dataset...")

    errors = []

    for row_number, row in df.iterrows():

        try:
            ds = json.loads(
                row["DeviceState"]
            )

            mw = json.loads(
                row["iot_middleware_data"]
            )

        except Exception as exc:

            errors.append(
                f"Row {row_number}: invalid JSON: {exc}"
            )

            continue

        # -----------------------------------
        # همه key ها وجود داشته باشند
        # -----------------------------------

        if set(ds.keys()) != set(ALL_FIELDS):

            errors.append(
                f"Row {row_number}: DeviceState schema mismatch"
            )

        if set(mw.keys()) != set(ALL_FIELDS):

            errors.append(
                f"Row {row_number}: middleware schema mismatch"
            )

        # -----------------------------------
        # ragImageURL همیشه null
        # -----------------------------------

        if ds["ragImageURL"] is not None:

            errors.append(
                f"Row {row_number}: DeviceState ragImageURL not null"
            )

        if mw["ragImageURL"] is not None:

            errors.append(
                f"Row {row_number}: middleware ragImageURL not null"
            )

        scenario = row["type"]

        # -----------------------------------
        # GET نباید write داشته باشد
        # -----------------------------------

        if scenario.startswith("get"):

            for field in ALL_FIELDS:

                if field in [
                    "message",
                    "ragImageURL",
                ]:
                    continue

                if mw[field] is not None:

                    errors.append(
                        f"Row {row_number}: GET attempted write to {field}"
                    )

        # -----------------------------------
        # vague نباید write داشته باشد
        # -----------------------------------

        if scenario == "vague":

            for field in ALL_FIELDS:

                if field in [
                    "message",
                    "ragImageURL",
                ]:
                    continue

                if ds[field] is not None:

                    errors.append(
                        f"Row {row_number}: vague DeviceState has {field}"
                    )

                if mw[field] is not None:

                    errors.append(
                        f"Row {row_number}: vague middleware has {field}"
                    )

        # -----------------------------------
        # invalid SET نباید write کند
        # -----------------------------------

        if scenario in [
            "set_invalid",
            "set_invalid_boundary",
            "set_read_only_rejected",
        ]:

            for field in ALL_FIELDS:

                if field in [
                    "message",
                    "ragImageURL",
                ]:
                    continue

                if mw[field] is not None:

                    errors.append(
                        f"Row {row_number}: unsafe write to {field}"
                    )

        # -----------------------------------
        # read-only ها هرگز write نشوند
        # -----------------------------------

        for read_only in READ_ONLY_PARAMETERS:

            if mw[read_only] is not None:

                errors.append(
                    f"Row {row_number}: write to read-only {read_only}"
                )

    if errors:

        print(
            f"Validation FAILED: {len(errors)} errors"
        )

        for error in errors[:30]:
            print(error)

        raise ValueError(
            "Dataset validation failed."
        )

    print(
        "Validation passed. No structural/safety errors found."
    )


# ============================================================
# ساخت dataset
# ============================================================

def build_dataset():

    rows.clear()

    # ترکیب پارامترهای قابل تنظیم
    generate_parameter_combinations()

    # boundary
    generate_temperature_boundary_cases()

    # read-only GET
    if GENERATE_READ_ONLY_GETS:

        for param in READ_ONLY_PARAMETERS:

            generate_get_case(
                (param,)
            )

    # GET ترکیبی با read-only
    generate_mixed_readonly_get_cases()

    # تلاش برای write به read-only
    generate_read_only_set_attempts()

    # alias
    generate_alias_get_cases()

    # smart mode semantic
    generate_smart_mode_semantic_cases()

    # vague
    generate_vague_cases()

    # unrelated
    generate_irrelevant_cases()

    df = pd.DataFrame(
        rows,
        columns=[
            "index",
            "DeviceState",
            "type",
            "iot_middleware_data",
        ],
    )

    return df


# ============================================================
# اجرا
# ============================================================

if __name__ == "__main__":

    df = build_dataset()

    validate_dataset(df)

    print("\nTotal rows:", len(df))

    print("\nScenario counts:")
    print(
        df["type"].value_counts()
    )

    print("\nFirst 5 rows:")
    print(
        df.head().to_string()
    )

    # CSV
    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # # JSONL
    # df.to_json(
    #     OUTPUT_JSONL,
    #     orient="records",
    #     lines=True,
    #     force_ascii=False,
    # )

    print(
        f"\nSaved CSV to: "
        f"{Path(OUTPUT_CSV).resolve()}"
    )

    # print(
    #     f"Saved JSONL to: "
    #     f"{Path(OUTPUT_JSONL).resolve()}"
    # )