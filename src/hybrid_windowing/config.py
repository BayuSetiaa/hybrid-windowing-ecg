"""Constants shared by the windowing pipeline and the experiments."""

WINDOW_SIZE = 360
BASE_STRIDE = 288
POOL_METHOD = "max"
GROUP_SIZE = 3
GROUP_ORDER_SEED = 42

RPEAK_SHIFT = 36
RPEAK_BASE_STRIDE = 72

EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 1.5e-3
WEIGHT_DECAY = 1e-4
SEEDS = (0, 1, 2, 3, 4)
INTRA_SEEDS = (0, 1, 2)
GRID_POOLS = ("max", "mean", "min")
GRID_GROUPS = (2, 3, 4)

CLASSES = ("N", "S", "V", "F")
CLASS_INDEX = {name: index for index, name in enumerate(CLASSES)}
AAMI_MAP = {
    **{symbol: "N" for symbol in "NLRej"},
    **{symbol: "S" for symbol in "AaJS"},
    "V": "V",
    "E": "V",
    "F": "F",
}

PACED_RECORDS = ("102", "104", "107", "217")
TRAIN_RECORDS = tuple("101 106 108 109 114 115 116 118 119 122 124 201 203 208 209 215 220 230".split())
VALIDATION_RECORDS = tuple("112 205 207 223".split())
TEST_RECORDS = tuple(
    "100 103 105 111 113 117 121 123 200 202 210 212 213 214 219 221 222 228 231 232 233 234".split()
)
ALL_RECORDS = tuple(sorted(TRAIN_RECORDS + VALIDATION_RECORDS + TEST_RECORDS))

CV_FOLDS = (
    {
        "train": "100 105 106 109 112 113 114 115 116 119 121 122 123 124 200 201 202 203 207 208 "
        "209 212 213 214 215 219 221 222 228 230 231 234",
        "validation": "118 220 223 233",
        "test": "101 103 108 111 117 205 210 232",
    },
    {
        "train": "100 101 103 105 106 108 109 111 113 115 116 117 118 119 121 124 200 203 205 207 "
        "210 212 213 214 219 221 222 223 228 232 234",
        "validation": "201 209 215 233",
        "test": "112 114 122 123 202 208 220 230 231",
    },
    {
        "train": "101 103 105 106 108 111 114 115 116 117 119 122 123 200 201 202 203 205 207 208 "
        "210 212 215 219 220 221 222 228 230 231 232",
        "validation": "112 209 223 233",
        "test": "100 109 113 118 121 124 213 214 234",
    },
    {
        "train": "100 101 103 105 106 108 109 111 112 113 114 117 119 121 122 123 124 201 202 205 "
        "208 210 213 214 215 219 220 221 230 231 232",
        "validation": "118 222 233 234",
        "test": "115 116 200 203 207 209 212 223 228",
    },
    {
        "train": "100 101 103 108 109 111 112 113 114 115 116 117 118 121 122 123 124 200 202 203 "
        "207 208 212 213 214 220 228 230 231 232 234",
        "validation": "205 209 210 223",
        "test": "105 106 119 201 215 219 221 222 233",
    },
)

EXPECTED_WINDOW_COUNTS = {
    "train_static": (36000, 431, 2605, 331),
    "train_hybrid": (12384, 3162, 7738, 1264),
    "validation_static": (8649, 120, 700, 23),
    "test_static": (42788, 1719, 2917, 382),
    "intra_static": (87437, 2270, 6222, 736),
    "intra_hybrid": (30028, 12765, 17093, 3181),
}
