import json
import pandas as pd
from typing import Dict, Any, List, Tuple
from datetime import datetime
import random
from itertools import combinations

class RefrigeratorDatasetGenerator:
    def __init__(self):
        self.test_data = []
        self.index_counter = 0
        
        # Parameter definitions
        self.numeric_ranges = {
            "refTemp": (1, 7),
            "frzTemp": (-24, -16),
        }
        
        self.enum_values = {
            "iceType": ["Crushed", "Cubed"],
            "smartMode": ["Default", "Power Saving", "Guest", "Night", "Novelty", "Off"],
        }
        
        self.boolean_params = [
            "supperRef", "supperFrz", "magicZone", "iceMaker", "light", 
            "childLock", "sleepMode", "vacation", "Hygiene", "doorAlarm", 
            "extraRef", "filterReset", "water", "ice_full", "iceFull"
        ]
        
        self.writable_params = [
            "refTemp", "frzTemp", "iceType", "smartMode", "supperRef", 
            "supperFrz", "magicZone", "iceMaker", "light", "childLock", 
            "sleepMode", "vacation", "Hygiene", "doorAlarm", "extraRef", 
            "filterReset", "water"
        ]
        
        self.readonly_params = ["iceFull", "filterTime", "environmentTemp"]
        
        # Questions in Persian for realism
        self.single_param_questions = {
            "refTemp": {
                "get": [
                    "دمای یخچال چند درجه است؟",
                    "درجه حرارت یخچال رو بگو",
                    "تنظیم دمای فریجیدر چند است؟",
                    "What is the refrigerator temperature?",
                    "Tell me the fridge temp",
                ],
                "set": [
                    ("دمای یخچال رو روی 3 درجه تنظیم کن", 3, True),
                    ("یخچال رو به 5 درجه ببر", 5, True),
                    ("تنظیم دما به 2", 2, True),
                    ("Set fridge to 1 degree", 1, True),
                    ("دمای یخچال 10 درجه", 10, False),  # Out of range
                    ("Temperature -5 for fridge", -5, False),  # Out of range
                    ("Set refrigerator to 0 degrees", 0, False),  # Out of range
                    ("دمای یخچال رو 8 درجه کن", 8, False),  # Out of range
                ],
                "irrelevant_get": [
                    "یخچال کی خریده شده؟",
                    "رنگ یخچال چیه؟",
                    "قیمت این یخچال چند بود؟",
                ]
            },
            "frzTemp": {
                "get": [
                    "دمای فریزر چند درجه است؟",
                    "تنظیمات فریزر رو بگو",
                    "دمای منجمد شدن چقدره؟",
                    "What is the freezer temperature?",
                    "Tell me freezer temp",
                ],
                "set": [
                    ("فریزر رو روی -16 درجه کن", -16, True),
                    ("دمای فریزر -20", -20, True),
                    ("Set freezer to -24 degrees", -24, True),
                    ("تنظیم فریزر به -18", -18, True),
                    ("فریزر رو 0 درجه کن", 0, False),  # Out of range
                    ("دمای فریزر 10 درجه", 10, False),  # Out of range
                    ("Freezer temperature -30", -30, False),  # Out of range
                ],
                "irrelevant_get": [
                    "فریزر چند سال گذاشته شده؟",
                    "فریزر چقدر جا دارد؟",
                    "آب فریزر رو کجا بریزم؟",
                ]
            },
            "iceType": {
                "get": [
                    "نوع یخ چیه؟",
                    "دستگاه یخ ساز چه نوع یخ درست می کنه؟",
                    "ما الآن اي نوع من الثلج تستخدم؟",
                    "What type of ice is being made?",
                ],
                "set": [
                    ("نوع یخ رو crushed کن", "Crushed", True),
                    ("یخ رو cubed تنظیم کن", "Cubed", True),
                    ("تغییر به crushed ice", "Crushed", True),
                    ("Change ice to cubed", "Cubed", True),
                    ("نوع یخ powdered", "Crushed", True),  # Synonym mapping
                    ("یخ رو dice کن", "Cubed", True),  # Synonym mapping
                    ("نوع یخ رو granulated کن", None, False),  # Invalid type
                ],
                "irrelevant_get": [
                    "یخ چند وقت طول می کشه درست شه؟",
                    "یخ خنک تر از فریزر است؟",
                ]
            },
            "smartMode": {
                "get": [
                    "هوشمند حالت چیه؟",
                    "الآن کدام smart mode فعال است؟",
                    "Mode دستگاه رو بگو",
                    "What is the current smart mode?",
                    "Tell me the device mode",
                ],
                "set": [
                    ("Smart mode رو Power Saving کن", "Power Saving", True),
                    ("Mode رو Night تنظیم کن", "Night", True),
                    ("Guest mode فعال کن", "Guest", True),
                    ("Switch to Novelty mode", "Novelty", True),
                    ("Set to Default", "Default", True),
                    ("Turn off smart mode", "Off", True),
                    ("Mode رو Cool کن", None, False),  # Invalid mode
                    ("Smart mode Ultra", None, False),  # Invalid mode
                ],
                "irrelevant_get": [
                    "Smart mode کی اضافه شده؟",
                    "Smart mode بیشتر برق می خوره؟",
                ]
            },
            "supperRef": {
                "get": [
                    "Super cool یخچال فعال است؟",
                    "Fast cooling refrigerator on؟",
                    "سوپر کول فعال هست؟",
                    "Is super cool mode active?",
                ],
                "set": [
                    ("سوپر کول یخچال رو فعال کن", True, True),
                    ("Fast cooling turn on", True, True),
                    ("Super cool refrigerator on", True, True),
                    ("سوپر کول رو خاموش کن", False, True),
                    ("Disable super cool", False, True),
                    ("Fast cooling off", False, True),
                ],
                "irrelevant_get": [
                    "سوپر کول چقدر برق می خوره؟",
                    "سوپر کول موقتیه؟",
                ]
            },
            "supperFrz": {
                "get": [
                    "Super freeze فعال است؟",
                    "Fast freezing on؟",
                    "Super freeze active؟",
                    "Is fast freezing active?",
                ],
                "set": [
                    ("Super freeze رو turn on کن", True, True),
                    ("Fast freezing فعال کن", True, True),
                    ("Freeze boost on", True, True),
                    ("Super freeze off", False, True),
                    ("Fast freezing خاموش کن", False, True),
                ],
                "irrelevant_get": [
                    "Super freeze چند درجه برودتی ایجاد می کنه؟",
                    "Super freeze دیگه یخ رو خراب می کنه؟",
                ]
            },
            "magicZone": {
                "get": [
                    "Magic zone فعال است؟",
                    "Chiller zone on؟",
                    "Fresh box active؟",
                    "Is the freshness box enabled?",
                ],
                "set": [
                    ("Magic zone رو فعال کن", True, True),
                    ("Chiller zone on", True, True),
                    ("Freshness box فعال کن", True, True),
                    ("Magic zone off", False, True),
                    ("Fresh box خاموش کن", False, True),
                ],
                "irrelevant_get": [
                    "Magic zone چی ذخیره می کنه؟",
                    "Magic zone چند درجه است؟",
                ]
            },
            "iceMaker": {
                "get": [
                    "دستگاه یخ ساز فعال است؟",
                    "Ice maker on؟",
                    "یخ ساز خاموش است؟",
                    "Is the ice maker active?",
                ],
                "set": [
                    ("یخ ساز رو فعال کن", True, True),
                    ("Ice maker turn on", True, True),
                    ("دستگاه یخ ساز on", True, True),
                    ("یخ ساز رو خاموش کن", False, True),
                    ("Ice maker off", False, True),
                ],
                "irrelevant_get": [
                    "یخ ساز هر چند وقت یخ درست می کنه؟",
                    "یخ ساز چقدر مصرف برق می کنه؟",
                ]
            },
            "light": {
                "get": [
                    "چراغ درب یخچال روشن است؟",
                    "Light on؟",
                    "روشنایی فعال هست؟",
                    "Is the refrigerator light on?",
                ],
                "set": [
                    ("چراغ یخچال رو روشن کن", True, True),
                    ("Light on", True, True),
                    ("روشنایی فعال کن", True, True),
                    ("چراغ خاموش کن", False, True),
                    ("Turn off light", False, True),
                ],
                "irrelevant_get": [
                    "چراغ چه رنگی است؟",
                    "چراغ کی خراب شدہ؟",
                ]
            },
            "childLock": {
                "get": [
                    "Child lock فعال است؟",
                    "بچه محافظ کننده فعال؟",
                    "Safety lock on؟",
                    "Is child lock active?",
                ],
                "set": [
                    ("Child lock رو فعال کن", True, True),
                    ("بچه محافظ فعال کن", True, True),
                    ("Safety lock turn on", True, True),
                    ("Child lock خاموش کن", False, True),
                    ("بچه محافظ off", False, True),
                ],
                "irrelevant_get": [
                    "Child lock چجوری کار می کنه؟",
                    "Child lock چند سال عمر داره؟",
                ]
            },
            "sleepMode": {
                "get": [
                    "Sleep mode فعال است؟",
                    "حالت خواب on؟",
                    "Night quiet mode active؟",
                    "Is sleep mode enabled?",
                ],
                "set": [
                    ("Sleep mode رو فعال کن", True, True),
                    ("حالت خواب on کن", True, True),
                    ("Night mode فعال کن", True, True),
                    ("Sleep mode خاموش کن", False, True),
                    ("حالت خواب off", False, True),
                ],
                "irrelevant_get": [
                    "Sleep mode چقدر صدای کم می کنه؟",
                    "Sleep mode تاثیری بر سرمایش داره؟",
                ]
            },
            "vacation": {
                "get": [
                    "Vacation mode فعال است؟",
                    "Eco mode on؟",
                    "Holiday mode active؟",
                    "Is vacation mode enabled?",
                ],
                "set": [
                    ("Vacation mode رو فعال کن", True, True),
                    ("Eco mode on کن", True, True),
                    ("Holiday mode فعال کن", True, True),
                    ("Vacation mode off", False, True),
                    ("Eco mode خاموش کن", False, True),
                ],
                "irrelevant_get": [
                    "Vacation mode برای چه مواقعی است؟",
                    "Vacation mode برق رو چقدر کم می کنه؟",
                ]
            },
            "Hygiene": {
                "get": [
                    "Hygiene filter فعال است؟",
                    "Air purifier on؟",
                    "Air filter active؟",
                    "Is the hygiene system enabled?",
                ],
                "set": [
                    ("Hygiene filter رو فعال کن", True, True),
                    ("Air purifier turn on", True, True),
                    ("Air filter on", True, True),
                    ("Hygiene filter خاموش کن", False, True),
                    ("Air purifier off", False, True),
                ],
                "irrelevant_get": [
                    "Hygiene filter چند وقت عمر داره؟",
                    "Hygiene filter چقدر خوبه؟",
                ]
            },
            "doorAlarm": {
                "get": [
                    "Door alarm فعال است؟",
                    "درب هشدار on؟",
                    "Alarm active؟",
                    "Is the door alarm enabled?",
                ],
                "set": [
                    ("Door alarm رو فعال کن", True, True),
                    ("درب هشدار on کن", True, True),
                    ("Alarm فعال کن", True, True),
                    ("Door alarm خاموش کن", False, True),
                    ("درب هشدار off", False, True),
                ],
                "irrelevant_get": [
                    "Door alarm چقدر خلوت است؟",
                    "Door alarm چند بار می ترجه؟",
                ]
            },
            "extraRef": {
                "get": [
                    "Extra cooling فعال است؟",
                    "Boost cooling on؟",
                    "Extra refrigeration active؟",
                    "Is extra cooling enabled?",
                ],
                "set": [
                    ("Extra cooling رو فعال کن", True, True),
                    ("Boost cooling on", True, True),
                    ("Extra refrigeration فعال کن", True, True),
                    ("Extra cooling off", False, True),
                    ("Boost cooling خاموش کن", False, True),
                ],
                "irrelevant_get": [
                    "Extra cooling کل برق رو چقدر افزایش می ده؟",
                    "Extra cooling توجه داره؟",
                ]
            },
            "filterReset": {
                "get": [
                    "Water filter reset شده؟",
                    "Filter counter reset on؟",
                    "Has the filter been reset?",
                ],
                "set": [
                    ("Water filter رو reset کن", True, True),
                    ("Filter counter reset", True, True),
                    ("فیلتر رو reset کن", True, True),
                ],
                "irrelevant_get": [
                    "Filter reset خودکار هست؟",
                    "Filter reset چند بار می شه؟",
                ]
            },
            "water": {
                "get": [
                    "Water dispenser فعال است؟",
                    "آب دادن on؟",
                    "Is water dispenser active?",
                ],
                "set": [
                    ("Water dispenser رو فعال کن", True, True),
                    ("آب دادن on کن", True, True),
                    ("Water on", True, True),
                    ("Water dispenser خاموش کن", False, True),
                    ("آب دادن off", False, True),
                ],
                "irrelevant_get": [
                    "Water dispenser چنده یخ بیرون می آره؟",
                    "آب دادن چقدر صاف است؟",
                ]
            },
        }
        
        # Vague questions
        self.vague_questions = [
            "سلام",
            "چطور می تونی کمکم کنی؟",
            "یخچالم رو میشه تغییر بدید؟",
            "چیزی برای بهتر شدن دارید؟",
            "یخچالم خوب کار می کنه؟",
            "Hello",
            "Hi there",
            "Can you help?",
            "What can you do?",
            "میشه یخچال رو کنترل کنیم؟",
        ]

    def get_default_device_state(self) -> Dict[str, Any]:
        """Generate a default device state"""
        return {
            "message": "",
            "refTemp": random.randint(1, 7),
            "frzTemp": random.randint(-24, -16),
            "iceType": random.choice(self.enum_values["iceType"]),
            "smartMode": random.choice([None] + self.enum_values["smartMode"]),
            "supperRef": random.choice([True, False]),
            "supperFrz": random.choice([True, False]),
            "magicZone": random.choice([True, False]),
            "iceMaker": random.choice([True, False]),
            "light": random.choice([True, False]),
            "childLock": random.choice([True, False]),
            "sleepMode": random.choice([True, False]),
            "vacation": random.choice([True, False]),
            "Hygiene": random.choice([True, False]),
            "doorAlarm": random.choice([True, False]),
            "extraRef": random.choice([True, False]),
            "filterReset": False,
            "water": random.choice([True, False]),
            "iceFull": random.choice([True, False]),
            "filterTime": random.randint(0, 500),
            "environmentTemp": random.randint(15, 35),
            "ragImageURL": None
        }

    def create_get_scenario(self, param: str, device_state: Dict[str, Any], 
                           question_text: str) -> Tuple[Dict, Dict]:
        """Create GET scenario data"""
        # Only include message and queried parameter in DeviceState
        dev_state = {key: None for key in device_state.keys()}
        dev_state["message"] = question_text
        dev_state[param] = device_state[param]
        
        # Set defaults for read-only params if they're being queried
        if param in self.readonly_params:
            dev_state[param] = device_state[param]
        
        # Preserve non-null values from original state for read-only params
        for readonly in self.readonly_params:
            if readonly in dev_state and dev_state[readonly] is None:
                dev_state[readonly] = device_state[readonly]
        
        # In iot_middleware_data, only message is populated with LLM response
        iot_data = {key: None for key in device_state.keys()}
        iot_data["message"] = self._generate_get_response(param, device_state[param])
        
        return dev_state, iot_data

    def create_set_scenario(self, param: str, device_state: Dict[str, Any], 
                           question_text: str, requested_value: Any, 
                           is_valid: bool) -> Tuple[Dict, Dict]:
        """Create SET scenario data"""
        # DeviceState has message and the parameter being set
        dev_state = {key: None for key in device_state.keys()}
        dev_state["message"] = question_text
        dev_state[param] = requested_value if is_valid else device_state[param]
        
        # Preserve readonly param values
        for readonly in self.readonly_params:
            dev_state[readonly] = device_state[readonly]
        
        # iot_middleware_data has message and parameter set to new value if valid
        iot_data = {key: None for key in device_state.keys()}
        
        if is_valid:
            iot_data["message"] = self._generate_set_success_response(param, requested_value)
            iot_data[param] = requested_value
        else:
            iot_data["message"] = self._generate_set_failure_response(param, requested_value)
            iot_data[param] = None
        
        return dev_state, iot_data

    def create_vague_scenario(self, device_state: Dict[str, Any], 
                             question_text: str) -> Tuple[Dict, Dict]:
        """Create VAGUE scenario data"""
        # DeviceState has only message
        dev_state = {key: None for key in device_state.keys()}
        dev_state["message"] = question_text
        
        # Preserve readonly param values
        for readonly in self.readonly_params:
            dev_state[readonly] = device_state[readonly]
        
        # iot_middleware_data has only clarification message
        iot_data = {key: None for key in device_state.keys()}
        iot_data["message"] = self._generate_vague_response()
        
        return dev_state, iot_data

    def _generate_get_response(self, param: str, value: Any) -> str:
        """Generate LLM response for GET scenario"""
        responses = {
            "refTemp": f"دمای فعلی یخچال {value} درجه سانتی‌گراد است.",
            "frzTemp": f"دمای فریزر {value} درجه سانتی‌گراد است.",
            "iceType": f"نوع یخ فعلی {value} است.",
            "smartMode": f"حالت هوشمند فعلی {value} است." if value else "حالتی هوشمند فعال نیست.",
            "supperRef": f"Super cool یخچال {'فعال' if value else 'غیرفعال'} است.",
            "supperFrz": f"Super freeze {'فعال' if value else 'غیرفعال'} است.",
            "magicZone": f"Magic zone {'فعال' if value else 'غیرفعال'} است.",
            "iceMaker": f"دستگاه یخ ساز {'فعال' if value else 'غیرفعال'} است.",
            "light": f"چراغ یخچال {'روشن' if value else 'خاموش'} است.",
            "childLock": f"Child lock {'فعال' if value else 'غیرفعال'} است.",
            "sleepMode": f"حالت خواب {'فعال' if value else 'غیرفعال'} است.",
            "vacation": f"Vacation mode {'فعال' if value else 'غیرفعال'} است.",
            "Hygiene": f"سیستم تصفیه هوا {'فعال' if value else 'غیرفعال'} است.",
            "doorAlarm": f"هشدار درب {'فعال' if value else 'غیرفعال'} است.",
            "extraRef": f"Extra cooling {'فعال' if value else 'غیرفعال'} است.",
            "water": f"Water dispenser {'فعال' if value else 'غیرفعال'} است.",
            "iceFull": f"سبد یخ {'پر' if value else 'خالی'} است.",
            "filterTime": f"فیلتر آب {value} ساعت استفاده شده است.",
            "environmentTemp": f"دمای محیط اطراف {value} درجه سانتی‌گراد است.",
        }
        return responses.get(param, f"اطلاعات {param} درخواست شده است.")

    def _generate_set_success_response(self, param: str, value: Any) -> str:
        """Generate LLM response for successful SET scenario"""
        responses = {
            "refTemp": f"دمای یخچال با موفقیت به {value} درجه تنظیم شد.",
            "frzTemp": f"دمای فریزر با موفقیت به {value} درجه تنظیم شد.",
            "iceType": f"نوع یخ با موفقیت به {value} تغییر کرد.",
            "smartMode": f"حالت هوشمند با موفقیت به {value} تنظیم شد.",
            "supperRef": f"Super cool یخچال با موفقیت {'فعال' if value else 'غیرفعال'} شد.",
            "supperFrz": f"Super freeze با موفقیت {'فعال' if value else 'غیرفعال'} شد.",
            "magicZone": f"Magic zone با موفقیت {'فعال' if value else 'غیرفعال'} شد.",
            "iceMaker": f"دستگاه یخ ساز با موفقیت {'فعال' if value else 'غیرفعال'} شد.",
            "light": f"چراغ یخچال با موفقیت {'روشن' if value else 'خاموش'} شد.",
            "childLock": f"Child lock با موفقیت {'فعال' if value else 'غیرفعال'} شد.",
            "sleepMode": f"حالت خواب با موفقیت {'فعال' if value else 'غیرفعال'} شد.",
            "vacation": f"Vacation mode با موفقیت {'فعال' if value else 'غیرفعال'} شد.",
            "Hygiene": f"سیستم تصفیه هوا با موفقیت {'فعال' if value else 'غیرفعال'} شد.",
            "doorAlarm": f"هشدار درب با موفقیت {'فعال' if value else 'غیرفعال'} شد.",
            "extraRef": f"Extra cooling با موفقیت {'فعال' if value else 'غیرفعال'} شد.",
            "filterReset": f"فیلتر آب با موفقیت reset شد.",
            "water": f"Water dispenser با موفقیت {'فعال' if value else 'غیرفعال'} شد.",
        }
        return responses.get(param, "تنظیمات با موفقیت اعمال شد.")

    def _generate_set_failure_response(self, param: str, value: Any) -> str:
        """Generate LLM response for failed SET scenario"""
        responses = {
            "refTemp": f"دمای {value} درجه برای یخچال معتبر نیست. لطفا مقدار بین 1 تا 7 درجه انتخاب کنید.",
            "frzTemp": f"دمای {value} درجه برای فریزر معتبر نیست. لطفا مقدار بین -24 تا -16 درجه انتخاب کنید.",
            "iceType": f"نوع یخ {value} معتبر نیست. لطفا Crushed یا Cubed انتخاب کنید.",
            "smartMode": f"حالت {value} معتبر نیست. لطفا یکی از حالت‌های معتبر انتخاب کنید.",
        }
        return responses.get(param, "این تنظیم قابل اعمال نیست.")

    def _generate_vague_response(self) -> str:
        """Generate clarification message for vague queries"""
        responses = [
            "متأسفانه سؤال شما برایم واضح نیست. آیا می‌خواهید دمای یخچال یا فریزر را تنظیم کنید؟ یا اطلاعاتی درباره تنظیمات دستگاه می‌خواهید؟",
            "لطفا بیشتر توضیح دهید. چه تغییری در یخچال می‌خواهید انجام دهید؟",
            "متأسفانه فهمیدم. آیا می‌تواند دوباره سؤال خود را بیان کنید؟ برای مثال: تنظیم دما، فعال کردن یک امکانت، یا درخواست اطلاعات؟",
            "I'm not sure what you're asking. Could you please clarify? Do you want to adjust temperature, enable/disable a feature, or get information about your refrigerator?",
        ]
        return random.choice(responses)

    def generate_single_parameter_tests(self) -> None:
        """Generate tests for single parameters"""
        device_state = self.get_default_device_state()
        
        for param in self.writable_params:
            if param not in self.single_param_questions:
                continue
            
            questions_dict = self.single_param_questions[param]
            
            # GET questions
            if "get" in questions_dict:
                for question in questions_dict["get"]:
                    dev_state, iot_data = self.create_get_scenario(
                        param, device_state, question
                    )
                    self.test_data.append({
                        "index": self.index_counter,
                        "DeviceState": json.dumps(dev_state, ensure_ascii=False),
                        "type": "get",
                        "iot_middleware_data": json.dumps(iot_data, ensure_ascii=False),
                    })
                    self.index_counter += 1
            
            # Irrelevant GET questions
            if "irrelevant_get" in questions_dict:
                for question in questions_dict["irrelevant_get"]:
                    _, iot_data = self.create_vague_scenario(device_state, question)
                    dev_state = {key: None for key in device_state.keys()}
                    dev_state["message"] = question
                    for readonly in self.readonly_params:
                        dev_state[readonly] = device_state[readonly]
                    
                    self.test_data.append({
                        "index": self.index_counter,
                        "DeviceState": json.dumps(dev_state, ensure_ascii=False),
                        "type": "vague",
                        "iot_middleware_data": json.dumps(iot_data, ensure_ascii=False),
                    })
                    self.index_counter += 1
            
            # SET questions
            if "set" in questions_dict:
                for question, value, is_valid in questions_dict["set"]:
                    dev_state, iot_data = self.create_set_scenario(
                        param, device_state, question, value, is_valid
                    )
                    self.test_data.append({
                        "index": self.index_counter,
                        "DeviceState": json.dumps(dev_state, ensure_ascii=False),
                        "type": "set",
                        "iot_middleware_data": json.dumps(iot_data, ensure_ascii=False),
                    })
                    self.index_counter += 1

    def generate_multi_parameter_tests(self) -> None:
        """Generate tests for multiple parameters"""
        device_state = self.get_default_device_state()
        
        # Generate combinations of 2, 3, and 4 parameters
        for combination_size in [2, 3, 4]:
            params = random.sample(self.writable_params, 
                                  min(combination_size, len(self.writable_params)))
            
            for param_combo in combinations(params, 
                                           min(combination_size, len(params))):
                # SET with multiple params
                set_questions = {
                    (("refTemp", "frzTemp"),): "دمای یخچال رو 3 و فریزر رو -18 کن",
                    (("supperRef", "supperFrz"),): "هم سوپر کول یخچال و هم فریزر رو فعال کن",
                    (("childLock", "light"),): "Child lock رو فعال و چراغ رو خاموش کن",
                    (("smartMode", "vacation"),): "Smart mode رو Night و vacation mode رو فعال کن",
                }
                
                for param_key, question in set_questions.items():
                    if param_combo in param_key:
                        # Create multi-param set scenario
                        dev_state = {key: None for key in device_state.keys()}
                        dev_state["message"] = question
                        
                        iot_data = {key: None for key in device_state.keys()}
                        iot_data["message"] = "تمام تنظیمات درخواستی با موفقیت اعمال شدند."
                        
                        # Set the values based on param combo
                        for idx, param in enumerate(param_combo):
                            if param == "refTemp":
                                dev_state[param] = 3
                                iot_data[param] = 3
                            elif param == "frzTemp":
                                dev_state[param] = -18
                                iot_data[param] = -18
                            elif param in ["supperRef", "supperFrz", "childLock"]:
                                dev_state[param] = True
                                iot_data[param] = True
                            elif param == "light":
                                dev_state[param] = False
                                iot_data[param] = False
                            elif param == "smartMode":
                                dev_state[param] = "Night"
                                iot_data[param] = "Night"
                            elif param == "vacation":
                                dev_state[param] = True
                                iot_data[param] = True
                        
                        # Preserve readonly params
                        for readonly in self.readonly_params:
                            dev_state[readonly] = device_state[readonly]
                        
                        self.test_data.append({
                            "index": self.index_counter,
                            "DeviceState": json.dumps(dev_state, ensure_ascii=False),
                            "type": "set_multi",
                            "iot_middleware_data": json.dumps(iot_data, ensure_ascii=False),
                        })
                        self.index_counter += 1

    def generate_vague_tests(self) -> None:
        """Generate vague question scenarios"""
        device_state = self.get_default_device_state()
        
        for question in self.vague_questions:
            dev_state, iot_data = self.create_vague_scenario(device_state, question)
            self.test_data.append({
                "index": self.index_counter,
                "DeviceState": json.dumps(dev_state, ensure_ascii=False),
                "type": "vague",
                "iot_middleware_data": json.dumps(iot_data, ensure_ascii=False),
            })
            self.index_counter += 1

    def generate_readonly_param_tests(self) -> None:
        """Generate tests for read-only parameters"""
        device_state = self.get_default_device_state()
        
        readonly_questions = {
            "iceFull": [
                "سبد یخ پر است؟",
                "Is the ice basket full?",
                "سبد یخ خالی؟",
            ],
            "filterTime": [
                "فیلتر آب چند ساعت استفاده شده؟",
                "How many hours has the water filter been used?",
                "مدت استفاده فیلتر رو بگو",
            ],
            "environmentTemp": [
                "دمای محیط اطراف یخچال چند است؟",
                "What is the room temperature?",
                "دمای اتاق چقدره؟",
            ]
        }
        
        for param, questions in readonly_questions.items():
            for question in questions:
                dev_state, iot_data = self.create_get_scenario(
                    param, device_state, question
                )
                self.test_data.append({
                    "index": self.index_counter,
                    "DeviceState": json.dumps(dev_state, ensure_ascii=False),
                    "type": "get_readonly",
                    "iot_middleware_data": json.dumps(iot_data, ensure_ascii=False),
                })
                self.index_counter += 1

    def generate_edge_case_tests(self) -> None:
        """Generate edge case tests"""
        device_state = self.get_default_device_state()
        
        edge_cases = [
            # Temperature boundaries
            ("Set fridge to maximum temperature", "refTemp", 7, True),
            ("Set fridge to minimum temperature", "refTemp", 1, True),
            ("Set freezer to maximum", "frzTemp", -16, True),
            ("Set freezer to minimum", "frzTemp", -24, True),
            
            # Just outside boundaries
            ("دمای یخچال 0 درجه", "refTemp", 0, False),
            ("دمای یخچال 8 درجه", "refTemp", 8, False),
            ("دمای فریزر -15 درجه", "frzTemp", -15, False),
            ("دمای فریزر -25 درجه", "frzTemp", -25, False),
            
            # Toggle booleans multiple times
            ("فعال کن، خاموش کن، دوباره فعال کن", "childLock", True, True),
        ]
        
        for question, param, value, is_valid in edge_cases:
            dev_state, iot_data = self.create_set_scenario(
                param, device_state, question, value, is_valid
            )
            self.test_data.append({
                "index": self.index_counter,
                "DeviceState": json.dumps(dev_state, ensure_ascii=False),
                "type": "set_edge",
                "iot_middleware_data": json.dumps(iot_data, ensure_ascii=False),
            })
            self.index_counter += 1

    def generate_dataset(self) -> pd.DataFrame:
        """Generate complete dataset"""
        print("Generating single parameter tests...")
        self.generate_single_parameter_tests()
        
        print("Generating multi-parameter tests...")
        self.generate_multi_parameter_tests()
        
        print("Generating read-only parameter tests...")
        self.generate_readonly_param_tests()
        
        print("Generating vague question tests...")
        self.generate_vague_tests()
        
        print("Generating edge case tests...")
        self.generate_edge_case_tests()
        
        df = pd.DataFrame(self.test_data)
        return df

# Main execution
if __name__ == "__main__":
    generator = RefrigeratorDatasetGenerator()
    dataset = generator.generate_dataset()
    
    print(f"\nTotal test cases generated: {len(dataset)}")
    print(f"Test type distribution:\n{dataset['type'].value_counts()}")
    
    # Save to CSV
    output_file = "./data/claude-haiku-4-5-20251001/refrigerator_rag_test_dataset.csv"
    dataset.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nDataset saved to {output_file}")
    
    # Display sample
    print("\n" + "="*100)
    print("SAMPLE TEST CASES")
    print("="*100)
    
    for idx in range(min(5, len(dataset))):
        print(f"\n--- Test Case {idx + 1} (Type: {dataset.iloc[idx]['type']}) ---")
        device_state = json.loads(dataset.iloc[idx]['DeviceState'])
        iot_data = json.loads(dataset.iloc[idx]['iot_middleware_data'])
        
        print(f"User Message: {device_state['message']}")
        print(f"\nDeviceState (selected fields):")
        for k, v in device_state.items():
            if v is not None and k != 'message':
                print(f"  {k}: {v}")
        
        print(f"\nLLM Response: {iot_data['message']}")
        print(f"\niot_middleware_data (parameters to apply):")
        for k, v in iot_data.items():
            if v is not None and k != 'message' and k != 'ragImageURL':
                print(f"  {k}: {v}")