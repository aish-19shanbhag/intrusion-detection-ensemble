"""Project-wide constants: paths, dataset schema, and the attack taxonomy."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
METRICS_DIR = ROOT / "reports" / "metrics"

TRAIN_FILE = DATA_DIR / "KDDTrain.txt"
TEST_FILE = DATA_DIR / "KDDTest.txt"
MODEL_FILE = MODELS_DIR / "ids_ensemble.joblib"
BEST_PARAMS_FILE = METRICS_DIR / "best_params.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20      # hold-out share for the primary evaluation
CV_FOLDS = 3          # folds used for feature selection and tuning

# NSL-KDD has 41 features, a text label, and a difficulty score
FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
COLUMN_NAMES = FEATURE_COLUMNS + ["label", "difficulty"]
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

# The ~40 specific attack labels grouped into the 5 standard categories
ATTACK_CATEGORY_MAP = {
    "normal": "normal",
    # Denial of Service: overwhelm a host so it cannot serve legitimate users
    "neptune": "dos", "back": "dos", "land": "dos", "pod": "dos", "smurf": "dos",
    "teardrop": "dos", "apache2": "dos", "udpstorm": "dos", "processtable": "dos",
    "mailbomb": "dos", "worm": "dos",
    # Probe: scan the network to find targets and weaknesses
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
    "mscan": "probe", "saint": "probe",
    # Remote to Local: gain local access from a remote machine
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l", "phf": "r2l",
    "multihop": "r2l", "warezmaster": "r2l", "warezclient": "r2l", "spy": "r2l",
    "xlock": "r2l", "xsnoop": "r2l", "snmpguess": "r2l", "snmpgetattack": "r2l",
    "httptunnel": "r2l", "sendmail": "r2l", "named": "r2l",
    # User to Root: escalate from a normal account to root
    "buffer_overflow": "u2r", "loadmodule": "u2r", "rootkit": "u2r", "perl": "u2r",
    "sqlattack": "u2r", "xterm": "u2r", "ps": "u2r",
}
CLASS_NAMES = ["dos", "normal", "probe", "r2l", "u2r"]
CLASS_DESCRIPTIONS = {
    "dos": "Denial of Service",
    "normal": "Normal traffic",
    "probe": "Probe / scanning",
    "r2l": "Remote to Local",
    "u2r": "User to Root",
}
