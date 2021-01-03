
import os

# **************************************** global variables: BEGIN ****************************************
_1KB = 1024
_1MB = _1KB**2
FPS = 30


# Good for debugging beam pipelines
FORCE_DISABLE_MULTIPROCESSING = False


TMP_DIR = '/tmp'
VIDEO_INDEX_BASE = 'video_index-20120129'
VIDEO_INDEXES_ARCHIVE = VIDEO_INDEX_BASE+'.zip'
VIDEO_INDEXES_DIR = os.path.join(TMP_DIR, VIDEO_INDEX_BASE)
SELECTED_VIDEO_INDEX_PATH = os.path.join(VIDEO_INDEXES_DIR, 'files_by_video_name.csv')
DOCUMENT_ASL_CONSULTANT_DS_FNAME = 'document-asl-consultant-index.csv'
ASL_CONSULTANT_DS_FNAME = 'asl-consultant-index.csv'
VIDEO_DS_FNAME = 'video.csv'
UTTERANCE_DS_FNAME = 'utterance-index.csv'
UTTERANCE_VIDEO_DS_FNAME = 'utterance-video-index.csv'
UTTERANCE_TOKEN_DS_FNAME = 'utterance-token-index.csv'
VOCABULARY_DS_FNAME = 'vocabulary-index.csv'
CORPUS_BASE = 'ncslgr-xml'
CORPUS_ARCHIVE = CORPUS_BASE+'.zip'
CORPUS_DIR = os.path.join(TMP_DIR, CORPUS_BASE)
CORPUS_DS_FNAME = 'ncslgr-corpus-index.csv'
VIDEO_SEGMENT_DS_FNAME = 'video-segment-index.csv'
UTTERANCE_TOKEN_FRAME_DS_FNAME = 'utterance-token-frame-index.csv'



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
  'Filename'
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
  'SegmentSequence',
  'Filename',
  'URL'
]
SCHEMA_PK__VIDEO_SEGMENT_DS = [
  SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS[0],
  SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS[1],
  SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS[2],
  SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS[3]
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
  'UtteranceSequence',
  'CameraPerspective'
]
SCHEMA_PK__UTTERANCE_VIDEO_DS = [
  SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[0],
  SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[1],
  SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[2],
  SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[3]
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
  'UtteranceID',
  'TokenID',
  'TokenSequence',
  'FrameSequence',
  'ImageTensor'
]
SCHEMA_PK__UTTERANCE_TOKEN_FRAME_DS = [
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[0],
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[1],
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[2],
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[3]
]

SCHEMA_COL_NAMES__VOCABULARY_DS = [
  'TokenID',
  'Token'
]
SCHEMA_PK__VOCABULARY_DS = [SCHEMA_COL_NAMES__VOCABULARY_DS[0]]

# note that this "schema" assumes intimate knowledge of 'files_by_video_name.csv' layout (i.e. the column-name/order mappings in it)
SCHEMA_COL_NAMES__VIDEO_INDEX = [
  'filename', 
  'video_seq_id', 
  'perspective_cam_id', 
  'compressed_mov_url', 
  'uncompressed_avi_url', 
  'uncompressed_avi_mirror_1_url', 
  'uncompressed_avi_mirror_2_url'
]
SCHEMA_PK__VIDEO_INDEX = [SCHEMA_COL_NAMES__VIDEO_INDEX[0]]
# ********** SCHEMA-related (FIXED) globals: END **********


# the following globals are set or modified at runtime
DATA_ROOT_DIR = None
VIDEO_DIR = None
STICHED_VIDEO_FRAMES_DIR = None
CORPUS_DS_PATH = None
MAX_RAW_XML_B64_LEN = None
ASL_CONSULTANT_DS_PATH = None
VIDEO_DS_PATH = None
UTTERANCE_DS_PATH = None
UTTERANCE_VIDEO_DS_PATH = None
UTTERANCE_TOKEN_DS_PATH = None
VOCABULARY_DS_PATH = None
CORPUS_DS_PATH = None
DOCUMENT_ASL_CONSULTANT_DS_PATH = None
VIDEO_DS_PATH = None
VIDEO_SEGMENT_DS_PATH = None
UTTERANCE_DS_PATH = None
UTTERANCE_VIDEO_DS_PATH = None
UTTERANCE_TOKEN_DS_PATH = None
UTTERANCE_TOKEN_FRAME_DS_PATH = None
VOCABULARY_DS_PATH = None
MAX_DATA_FILES = None

D_IN_MEMORY_VARS = dict()
# **************************************** global variables: END ****************************************