def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    reverse = dict((value, key) for key, value in enums.items())  #enums.iteritems
    enums['reverse_mapping'] = reverse
    return type('Enum', (), enums)

DXF_LAYER = enum(ALPHA_LINEAR='alpha_linear', ALPHA_OTHER='alpha_other',
                 SIMPLIFIED_LINEAR='simplified_linear', SIMPLIFIED_OTHER='simplified_other',
                 MBR_LINEAR='mbr_linear', MBR_OTHER='mbr_other')

SEGMENT_SIDE = enum(BASE_SIDE = 0, RIGHT_SIDE = 1, OPPOSITE_SIDE = 2, LEFT_SIDE = 3)

ALPHA_SHAPE_MODE = enum('CGAL','POLYGONIZE','MYPOLYGON')

MACHINE_LEARNING_CLASSES = enum(BEAM_SEGMENT = 0, SPLIT_SEGMENT = 1, OTHER_SEGMENT = 2)

SHAPE_CLASSIFICATION = enum(UNCLASSIFIED = 0, BEAM_SEGMENT = 1, UI_DECISION = 2, OTHER_SEGMENT = 3)

BEAM_ORIENTATION = enum(DEFAULT=0, Z_POSITIVE=1, Z_NEGATIVE=-1)
BEAM_JOIN_MODE = enum(ELONGATE=0, TRIM=1, TRIM_AND_ELONGATE=2, ELONGATE_FIRST=3, TRIM_FIRST=4)

ELONGATION_MODE = enum(ELONGATE="Elongate existing beams", CREATE_NEW="Create new beams")