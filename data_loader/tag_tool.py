import dateutil.parser


rating_map = {
    "general": ["safe"],
    "sensitive": ["sensitive"],
    "questionable": ["nsfw"],
    "explicit": ["explicit", "nsfw"],
    "g": ["safe"],
    "s": ["sensitive"],
    "q": ["nsfw"],
    "e": ["explicit", "nsfw"],
}

special_tags = {
    "1girl",
    "2girls",
    "3girls",
    "4girls",
    "5girls",
    "6+girls",
    "multiple_girls",
    "1boy",
    "2boys",
    "3boys",
    "4boys",
    "5boys",
    "6+boys",
    "multiple_boys",
    "male_focus",
    "1other",
    "2others",
    "3others",
    "4others",
    "5others",
    "6+others",
    "multiple_others",
}

meta_keywords_black_list = [
    "(medium)",
    "commentary",
    "bad",
    "translat",
    "request",
    "mismatch",
    "revision",
    "audio",
    "video",
]


fav_count_percentile_full = {
    "general": {
        5: 1, 10: 1, 15: 2, 20: 3, 25: 3, 30: 4, 35: 5,
        40: 6, 45: 7, 50: 8, 55: 9, 60: 10, 65: 12,
        70: 14, 75: 16, 80: 18, 85: 22, 90: 27, 95: 37
    },
    "g": {
        5: 1, 10: 1, 15: 2, 20: 3, 25: 3, 30: 4, 35: 5,
        40: 6, 45: 7, 50: 8, 55: 9, 60: 10, 65: 12,
        70: 14, 75: 16, 80: 18, 85: 22, 90: 27, 95: 37
    },
    "sensitive": {
        5: 1, 10: 2, 15: 4, 20: 5, 25: 6, 30: 8, 35: 9,
        40: 11, 45: 13, 50: 15, 55: 17, 60: 19, 65: 22,
        70: 26, 75: 30, 80: 36, 85: 44, 90: 56, 95: 81
    },
    "s": {
        5: 1, 10: 2, 15: 4, 20: 5, 25: 6, 30: 8, 35: 9,
        40: 11, 45: 13, 50: 15, 55: 17, 60: 19, 65: 22,
        70: 26, 75: 30, 80: 36, 85: 44, 90: 56, 95: 81
    },
    "questionable": {
        5: 4, 10: 8, 15: 11, 20: 14, 25: 18, 30: 21, 35: 25,
        40: 29, 45: 33, 50: 38, 55: 43, 60: 49, 65: 56,
        70: 65, 75: 75, 80: 88, 85: 105, 90: 132, 95: 182
    },
    "q": {
        5: 4, 10: 8, 15: 11, 20: 14, 25: 18, 30: 21, 35: 25,
        40: 29, 45: 33, 50: 38, 55: 43, 60: 49, 65: 56,
        70: 65, 75: 75, 80: 88, 85: 105, 90: 132, 95: 182
    },
    "explicit": {
        5: 4, 10: 9, 15: 13, 20: 18, 25: 22, 30: 27, 35: 33,
        40: 39, 45: 45, 50: 52, 55: 60, 60: 69, 65: 79,
        70: 92, 75: 106, 80: 125, 85: 151, 90: 190, 95: 262
    },
    "e": {
        5: 4, 10: 9, 15: 13, 20: 18, 25: 22, 30: 27, 35: 33,
        40: 39, 45: 45, 50: 52, 55: 60, 60: 69, 65: 79,
        70: 92, 75: 106, 80: 125, 85: 151, 90: 190, 95: 262
    },
}

score_percentile_full = {
    "general": {
        5: 0,
        10: 1,
        15: 2,
        20: 3,
        25: 3,
        30: 4,
        35: 5,
        40: 5,
        45: 6,
        50: 7,
        55: 8,
        60: 9,
        65: 11,
        70: 12,
        75: 14,
        80: 16,
        85: 19,
        90: 24,
        95: 33,
    },
    "sensitive": {
        5: 0,
        10: 1,
        15: 2,
        20: 3,
        25: 4,
        30: 5,
        35: 6,
        40: 7,
        45: 8,
        50: 9,
        55: 11,
        60: 12,
        65: 15,
        70: 17,
        75: 20,
        80: 25,
        85: 31,
        90: 41,
        95: 62,
    },
    "questionable": {
        5: 2,
        10: 4,
        15: 5,
        20: 7,
        25: 9,
        30: 11,
        35: 14,
        40: 16,
        45: 19,
        50: 23,
        55: 26,
        60: 31,
        65: 36,
        70: 42,
        75: 49,
        80: 59,
        85: 73,
        90: 93,
        95: 134,
    },
    "explicit": {
        5: 2,
        10: 4,
        15: 7,
        20: 10,
        25: 13,
        30: 17,
        35: 20,
        40: 25,
        45: 29,
        50: 35,
        55: 41,
        60: 48,
        65: 56,
        70: 66,
        75: 78,
        80: 94,
        85: 115,
        90: 148,
        95: 211,
    },
    "g": {
        5: 0,
        10: 1,
        15: 2,
        20: 3,
        25: 3,
        30: 4,
        35: 5,
        40: 5,
        45: 6,
        50: 7,
        55: 8,
        60: 9,
        65: 11,
        70: 12,
        75: 14,
        80: 16,
        85: 19,
        90: 24,
        95: 33,
    },
    "s": {
        5: 0,
        10: 1,
        15: 2,
        20: 3,
        25: 4,
        30: 5,
        35: 6,
        40: 7,
        45: 8,
        50: 9,
        55: 11,
        60: 12,
        65: 15,
        70: 17,
        75: 20,
        80: 25,
        85: 31,
        90: 41,
        95: 62,
    },
    "q": {
        5: 2,
        10: 4,
        15: 5,
        20: 7,
        25: 9,
        30: 11,
        35: 14,
        40: 16,
        45: 19,
        50: 23,
        55: 26,
        60: 31,
        65: 36,
        70: 42,
        75: 49,
        80: 59,
        85: 73,
        90: 93,
        95: 134,
    },
    "e": {
        5: 2,
        10: 4,
        15: 7,
        20: 10,
        25: 13,
        30: 17,
        35: 20,
        40: 25,
        45: 29,
        50: 35,
        55: 41,
        60: 48,
        65: 56,
        70: 66,
        75: 78,
        80: 94,
        85: 115,
        90: 148,
        95: 211,
    },
}

score_percentile_after_5m = {
    "general": {
        5: 1,
        10: 1,
        15: 2,
        20: 3,
        25: 4,
        30: 5,
        35: 5,
        40: 6,
        45: 7,
        50: 8,
        55: 10,
        60: 11,
        65: 13,
        70: 14,
        75: 17,
        80: 19,
        85: 23,
        90: 28,
        95: 38,
    },
    "sensitive": {
        5: 2,
        10: 4,
        15: 7,
        20: 9,
        25: 11,
        30: 13,
        35: 16,
        40: 18,
        45: 21,
        50: 24,
        55: 28,
        60: 31,
        65: 36,
        70: 41,
        75: 48,
        80: 56,
        85: 67,
        90: 84,
        95: 118,
    },
    "questionable": {
        5: 4,
        10: 9,
        15: 14,
        20: 19,
        25: 24,
        30: 29,
        35: 34,
        40: 40,
        45: 45,
        50: 52,
        55: 59,
        60: 67,
        65: 75,
        70: 86,
        75: 98,
        80: 114,
        85: 136,
        90: 168,
        95: 228,
    },
    "explicit": {
        5: 5,
        10: 11,
        15: 16,
        20: 22,
        25: 28,
        30: 34,
        35: 40,
        40: 47,
        45: 54,
        50: 63,
        55: 72,
        60: 82,
        65: 94,
        70: 108,
        75: 124,
        80: 145,
        85: 174,
        90: 216,
        95: 296,
    },
    "g": {
        5: 1,
        10: 1,
        15: 2,
        20: 3,
        25: 4,
        30: 5,
        35: 5,
        40: 6,
        45: 7,
        50: 8,
        55: 10,
        60: 11,
        65: 13,
        70: 14,
        75: 17,
        80: 19,
        85: 23,
        90: 28,
        95: 38,
    },
    "s": {
        5: 2,
        10: 4,
        15: 7,
        20: 9,
        25: 11,
        30: 13,
        35: 16,
        40: 18,
        45: 21,
        50: 24,
        55: 28,
        60: 31,
        65: 36,
        70: 41,
        75: 48,
        80: 56,
        85: 67,
        90: 84,
        95: 118,
    },
    "q": {
        5: 4,
        10: 9,
        15: 14,
        20: 19,
        25: 24,
        30: 29,
        35: 34,
        40: 40,
        45: 45,
        50: 52,
        55: 59,
        60: 67,
        65: 75,
        70: 86,
        75: 98,
        80: 114,
        85: 136,
        90: 168,
        95: 228,
    },
    "e": {
        5: 5,
        10: 11,
        15: 16,
        20: 22,
        25: 28,
        30: 34,
        35: 40,
        40: 47,
        45: 54,
        50: 63,
        55: 72,
        60: 82,
        65: 94,
        70: 108,
        75: 124,
        80: 145,
        85: 174,
        90: 216,
        95: 296,
    },
}



def year_tag(
    created_at
) -> str:
    year = 0
    try:
        date = dateutil.parser.parse(created_at)
        year = date.year
    except:
        pass
    if 2005 <= year <= 2010:
        year_tag = "old"
    elif year <= 2014:
        year_tag = "early"
    elif year <= 2017:
        year_tag = "mid"
    elif year <= 2020:
        year_tag = "recent"
    elif year <= 2024:
        year_tag = "newest"
    else:
        return None

    return year_tag


def rating_tag(
    rating
) -> str:
    if (tag := rating_map.get(rating, None)) is not None:
        return tag[0]
    else:
        return None


def quality_tag(
    id,
    fav_count,
    rating,
    
    percentile_map: dict[str, dict[int, int]] = fav_count_percentile_full,
) -> tuple[list[str], list[str]]:
    if int(id) > 7800000:
        # Don't add quality tag for posts which are new.
        return None
    else:
        percentile_map = fav_count_percentile_full
        rating = rating
        score = fav_count
        percentile = percentile_map[rating]

        if score > percentile[95]:
            quality_tag = "masterpiece"
        elif score > percentile[85]:
            quality_tag = "best quality"
        elif score > percentile[75]:
            quality_tag = "great quality"
        elif score > percentile[50]:
            quality_tag = "good quality"
        elif score > percentile[25]:
            quality_tag = "normal quality"
        elif score > percentile[10]:
            quality_tag = "low quality"
        else:
            quality_tag = "worst quality"

        

        return quality_tag

def tags_filter(tag, black_list: list[str]) -> bool:
    return not any(keyword in tag for keyword in black_list)


def meta_tags_filter(tags) -> list[str]:


    # NOTE: we only filter out meta tags with these keywords
    # Which is definitely not related to the content of image
    return [tag for tag in tags if tags_filter(tag, meta_keywords_black_list)]

def extract_special_tags(tag_list) -> tuple[list[str], list[str]]:
    special = []
    general = []
    for tag in tag_list:
        if tag in special_tags:
            special.append(tag)
        else:
            general.append(tag)
    return special, general



    