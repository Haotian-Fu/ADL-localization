import numpy as np

nodes_info = {
    1: {
        "id": "b8-27-eb-63-ae-61",
        "range": [0, 60]
    },
    2: {
        "id": "b8-27-eb-4e-d2-eb",
        "range": [0, 188]
    },
    3: {
        "id": "b8-27-eb-dc-a9-b5",
        "range": [0, 70]
    },
    4: {
        "id": "b8-27-eb-b4-f8-c2",
        "range": [0, 170]
    },
    5: {
        "id": "b8-27-eb-02-d4-0b",
        "range": [0, 45]
    },
    6: {
        "id": "b8-27-eb-86-23-51",
        "range": [0, 130]
    },
    7: {
        "id": "b8-27-eb-cf-59-2a",
        "range": [0, 75]
    },
    8: {
        "id": "b8-27-eb-be-fd-bf",
        "range": [0, 75]
    },
    9: {
        "id": "b8-27-eb-92-28-87",
        "range": [0, 75]
    },
    10: {
        "id": "b8-27-eb-4f-46-d7",
        "range": [0, 100]
    },
    11: {
        "id": "b8-27-eb-3f-d0-0b",
        "range": [0, 100]
    },
    12: {
        "id": "b8-27-eb-a3-b3-2a",
        "range": [0, 100]
    },
    13: {
        "id": "b8-27-eb-85-a7-83",
        "range": [0, 130]
    },
    14: {
        "id": "b8-27-eb-1b-02-d2",
        "range": [0, 130]
    },
    15: {
        "id": "b8-27-eb-2c-3e-07",
        "range": [0, 130]
    },
    16: {
        "id": "b8-27-eb-ec-77-37",
        "range": [0, 188]
    }
}


nodes_info_pi4 = {
    1: {
        "id": "e4-5f-01-8a-e1-d4",
        "range": [0, 60]
    },
    2: {
        "id": "e4-5f-01-78-c9-40",
        "range": [0, 188]
    },
    3: {
        "id": "e4-5f-01-8a-e3-09",
        "range": [0, 70]
    },
    4: {
        "id": "e4-5f-01-8a-45-b5",
        "range": [0, 170]
    },
    5: {
        "id": "e4-5f-01-8b-08-5a",
        "range": [0, 45]
    },
    6: {
        "id": "e4-5f-01-8a-46-81",
        "range": [0, 130]
    },
    7: {
        "id": "e4-5f-01-88-59-21",
        "range": [0, 75]
    },
    8: {
        "id": "e4-5f-01-88-5b-ff",
        "range": [0, 75]
    },
    9: {
        "id": "e4-5f-01-8b-16-92",
        "range": [0, 75]
    },
    10: {
        "id": "dc-a6-32-f7-bd-39",
        "range": [0, 100]
    },
    11: {
        "id": "e4-5f-01-8b-2d-e2",
        "range": [0, 100]
    },
    12: {
        "id": "e4-5f-01-88-60-ff",
        "range": [0, 100]
    },
    13: {
        "id": "e4-5f-01-8a-df-32",
        "range": [0, 130]
    },
    14: {
        "id": "e4-5f-01-8b-18-b1",
        "range": [0, 130]
    },
    15: {
        "id": "e4-5f-01-8a-78-6f",
        "range": [0, 130]
    },
    16: {
        "id": "e4-5f-01-51-a7-97",
        "range": [0, 188]
    }
}

room_info = {   # for new ADL script
    "Kitchen": {
        "sensors": [2, 3, 5, 6, 13],
        "activities": np.arange(1, 10).tolist() + np.arange(38, 40).tolist()
    },
    "Couch": {
        "sensors": [14, 15, 16],
        "activities": np.arange(10, 15)
    },
    "Bathroom": {
        "sensors": [7, 8, 9],
        "activities": np.arange(15, 23)
    },
    "Bedroom": {
        "sensors": [10, 11, 12],
        "activities": np.arange(23, 34)
    },
    "Studytable": {
        "sensors": [14, 15, 16],
        "activities": np.arange(34, 38)
    },
    # "Livingroom": {
    #     "sensors": [1, 2, 4, 5, 13, 14, 15, 16],
    #     "activities": np.arange(1, 40) - np.arrange(2, 9)
    # },
}


# frame numer of each session, used for mmap data loading
# frame_num = {
#     "8F33UK": 38659,
#     "85XB4Y": 43592,
#     "SB-46951W": 45867,
#     "SB-50274X": 32348,
#     "SB-50274X-2": 29004,
# }
sessions = ["0exwAT_ADL_1", "1eKOIF_ADL_1", "6e5iYM_ADL_1", "8F33UK", "85XB4Y", "eg35Wb_ADL_1", "I2HSeJ_ADL_1", "NQHEKm_ADL_1",
            "rjvUbM_ADL_2", "RQAkB1_ADL_1", "SB-00834W", "SB-46951W", "SB-50274X", "SB-50274X-2", "SB-94975U",
            "SB-94975U-2", "YhsHv0_ADL_1", "YpyRw1_ADL_2", 
            # "SB-09565T"
            ]

