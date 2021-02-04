from __future__ import absolute_import

import os

# **************************************** global variables: BEGIN ****************************************
_1KB = 1024
_1MB = _1KB**2
FPS = 30
# FRAME_IMG_INPUT_SHAPE = (300,300)
FRAME_IMG_INPUT_SHAPE = (150,150)
MAX_CAMERA_PERSPECTIVES = 4
VALIDATION_SIZE_RATIO = .10

DOWNLOAD_MAX_FAIL_COUNT = 10
DOWNLOAD_FAIL_SLEEP_TIME = 1  # seconds


VALIDATION_WARNING_TEXT = "***VALIDATION WARNING!!!***:"
VALIDATION_FATAL_ERROR_TEXT = "***VALIDATION FATAL ERROR!!!***:"

OUTPUT_INFO_LEVEL__DEBUG = 0
OUTPUT_INFO_LEVEL__WARNING = 1
OUTPUT_INFO_LEVEL__ERROR = 2
OUTPUT_INFO_LEVEL = OUTPUT_INFO_LEVEL__WARNING


# Good for debugging beam pipelines
FORCE_DISABLE_MULTIPROCESSING = False


# TMP_DIR = '/tmp'

CORPUS_BASE = 'ncslgr-xml'
CORPUS_ARCHIVE = CORPUS_BASE+'.zip'
# CORPUS_DIR = os.path.join(TMP_DIR, CORPUS_BASE)
CORPUS_DS_FNAME = 'ncslgr-corpus-index.csv'

VIDEO_INDEX_BASE = 'video_index-20120129'
VIDEO_INDEXES_ARCHIVE = VIDEO_INDEX_BASE+'.zip'
# VIDEO_INDEXES_DIR = os.path.join(TMP_DIR, VIDEO_INDEX_BASE)
# SELECTED_VIDEO_INDEX_PATH = os.path.join(VIDEO_INDEXES_DIR, 'files_by_video_name.csv')

DOCUMENT_ASL_CONSULTANT_DS_FNAME = 'document-consultant-index.csv'
ASL_CONSULTANT_DS_FNAME = 'consultant-index.csv'

VIDEO_DS_FNAME = 'document-consultant-targetvideo-index.csv'
UTTERANCE_DS_FNAME = 'document-consultant-utterance-index.csv'

UTTERANCE_VIDEO_DS_FNAME = 'document-consultant-utterance-targetvideo-index.csv'
UTTERANCE_TOKEN_DS_FNAME = 'document-consultant-utterance-token-index.csv'

VIDEO_SEGMENT_DS_FNAME = 'document-consultant-targetvideo-segment-index.csv'
VIDEO_FRAME_DS_FNAME = 'document-consultant-targetvideo-frame-index.csv'
UTTERANCE_TOKEN_FRAME_DS_FNAME = 'document-consultant-targetvideo-utterance-token-frame-index.csv'

VOCABULARY_DS_FNAME = 'vocabulary-index.csv'

TRAIN_FRAME_SEQ_ASSOC_DS_FNAME = 'train-assoc.csv'
VAL_FRAME_SEQ_DS_FNAME = 'val.csv'
TRAIN_FRAME_SEQ_DS_FNAME = 'train.csv'

COMPLETE_UTTERANCES_TRAIN_ASSOC_DS_FNAME = 'complete-utterances-train-assoc.csv'
COMPLETE_UTTERANCES_VAL_DS_FNAME = 'complete-utterances-val.csv'
COMPLETE_UTTERANCES_TRAIN_DS_FNAME = 'complete-utterances-train.csv'



# ********** SCHEMA-related (FIXED) globals: BEGIN **********
SCHEMA_COL_NAMES__CORPUS_DS = [
  'DocumentID',
  'Filename',
  'XML_B64',
  'LEN'
]
SCHEMA_PK__CORPUS_DS = [SCHEMA_COL_NAMES__CORPUS_DS[0]]

SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS = [
  'DocumentID',
  'ASLConsultantID'
]
SCHEMA_PK__DOCUMENT_ASL_CONSULTANT_DS = [
  SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS[0], 
  SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS[1]
]

SCHEMA_COL_NAMES__ASL_CONSULTANT_DS = [
  'ASLConsultantID',
  'Name',
  'Age',
  'Gender'
]
SCHEMA_PK__ASL_CONSULTANT_DS = [SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[0]]

SCHEMA_COL_NAMES__VIDEO_DS = [
  'DocumentID',
  'ASLConsultantID',
  'CameraPerspective',
  'TargetVideoFilename'
]
SCHEMA_PK__VIDEO_DS = [
  SCHEMA_COL_NAMES__VIDEO_DS[0],
  SCHEMA_COL_NAMES__VIDEO_DS[1],
  SCHEMA_COL_NAMES__VIDEO_DS[2]
]

SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS = [
  'DocumentID',
  'ASLConsultantID',
  'CameraPerspective',
  'TargetVideoFilename',
  'SegmentSequence',
  'SegmentVideoFilename',
  'URL'
]
SCHEMA_PK__VIDEO_SEGMENT_DS = [
  SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS[0],
  SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS[1],
  SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS[2],
  SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS[3],
  SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS[4]
]

SCHEMA_COL_NAMES__VIDEO_FRAME_DS = [
  'DocumentID',
  'ASLConsultantID',
  'CameraPerspective',
  'TargetVideoFilename',
  'FrameSequence',
  'FramePath',
  # 'JPEGBytes'
]
SCHEMA_PK__VIDEO_FRAME_DS = [
  SCHEMA_COL_NAMES__VIDEO_FRAME_DS[0],
  SCHEMA_COL_NAMES__VIDEO_FRAME_DS[1],
  SCHEMA_COL_NAMES__VIDEO_FRAME_DS[2],
  SCHEMA_COL_NAMES__VIDEO_FRAME_DS[3],
  SCHEMA_COL_NAMES__VIDEO_FRAME_DS[4]
]


SCHEMA_COL_NAMES__UTTERANCE_DS = [
  'DocumentID',
  'ASLConsultantID',
  'UtteranceSequence',
  'StartTime',
  'EndTime',
  'Tokens',
  'Translation'
]
SCHEMA_PK__UTTERANCE_DS = [
  SCHEMA_COL_NAMES__UTTERANCE_DS[0],
  SCHEMA_COL_NAMES__UTTERANCE_DS[1],
  SCHEMA_COL_NAMES__UTTERANCE_DS[2]
]

SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS = [
  'DocumentID',
  'ASLConsultantID',
  'TargetVideoFilename',
  'UtteranceSequence',
  'CameraPerspective'
]
SCHEMA_PK__UTTERANCE_VIDEO_DS = [
  SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[0],
  SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[1],
  SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[2],
  SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[3],
  SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[4]
]

SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS = [
  'DocumentID',
  'ASLConsultantID',
  'UtteranceSequence',
  'TokenSequence',

  'StartTime',
  'EndTime',
  'TokenID',
  'Field',
  'FieldValue'
]
SCHEMA_PK__UTTERANCE_TOKEN_DS = [
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[0],
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[1],
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[2],
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[3]
]

SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS = [
  'DocumentID',
  'ASLConsultantID',
  'CameraPerspective',
  'TargetVideoFilename',
  'UtteranceSequence',
  'TokenSequence',
  'FrameSequence',

  'TokenID'
]
SCHEMA_PK__UTTERANCE_TOKEN_FRAME_DS = [
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[0],
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[1],
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[2],
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[3],
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[4],
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[5],
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[6]
]

SCHEMA_COL_NAMES__VOCABULARY_DS = [
  'TokenID',
  'Token'
]
SCHEMA_PK__VOCABULARY_DS = [SCHEMA_COL_NAMES__VOCABULARY_DS[0]]

# note that this "schema" assumes intimate knowledge of 'files_by_video_name.csv' layout (i.e. the column-name/order mappings in it)
SCHEMA_COL_NAMES__VIDEO_INDEX = [
  'target_video_filename', 
  'target_video_seq_id', 
  'perspective_cam_id', 
  'compressed_mov_url',             # has a list of segment video urls, delimetted by ';'
  'uncompressed_avi_url', 
  'uncompressed_avi_mirror_1_url', 
  'uncompressed_avi_mirror_2_url'
]
SCHEMA_PK__VIDEO_INDEX = [SCHEMA_COL_NAMES__VIDEO_INDEX[0]]

SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX = [
  'TokenID',
  'CameraPerspective',
  'ASLConsultantID',
  'TargetVideoFilename',
  'UtteranceSequence',
  'TokenSequence',
  'FrameSequence'
]
SCHEMA_PK__TRAIN_OR_VAL_INDEX = [
  SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[0],
  SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[1],
  SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[2],
  SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[3],
  SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[4],
  SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[5]
]

SCHEMA_COL_NAMES__COMPLETE_UTTERANCES_TRAIN_VAL_TCP_INDEX = [
    'ASLConsultantID',
    'TargetVideoFilename',
    'UtteranceSequence',
    'CameraPerspective',
    'StartTime',
    'EndTime',
    'TokenIDs',
    'Tokens',
    'Translation'
]
# ********** SCHEMA-related (FIXED) globals: END **********


# the following globals are set or modified at runtime
WORK_DIR = None
GCS_CLIENT = None
GCS_BUCKET = None
GCS_IO = None
DATA_ROOT_DIR = None
TMP_DIR = None
VIDEO_DIR = None
STICHED_VIDEO_FRAMES_DIR = None
CORPUS_DS_PATH = None
MAX_RAW_XML_B64_LEN = 1000000 # default
ASL_CONSULTANT_DS_PATH = None
VIDEO_DS_PATH = None
UTTERANCE_DS_PATH = None
UTTERANCE_VIDEO_DS_PATH = None
UTTERANCE_TOKEN_DS_PATH = None
CORPUS_DS_PATH = None
DOCUMENT_ASL_CONSULTANT_DS_PATH = None
VIDEO_SEGMENT_DS_PATH = None
VIDEO_FRAME_DS_PATH = None
UTTERANCE_TOKEN_FRAME_DS_PATH = None
VOCABULARY_DS_PATH = None
TRAIN_ASSOC_DS_PATH = None
VAL_DS_PATH = None
TRAIN_DS_PATH = None
COMPLETE_UTTERANCES_TRAIN_ASSOC_DS_PATH = None
COMPLETE_UTTERANCES_VAL_DS_PATH = None
COMPLETE_UTTERANCES_TRAIN_DS_PATH = None
MAX_TARGET_VIDEOS = None

D_IN_MEMORY_VARS = dict()
# **************************************** global variables: END ****************************************
