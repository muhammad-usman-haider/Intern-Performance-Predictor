CSV_PATH = "data/intern_performance_data.csv"
FEATURES = ['attendance_rate', 'task_completion_rate', 'avg_feedback_score', 'final_assessment_score']
TARGET = 'performance_label'
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS = 5  # Stratified K-Fold
FIG_DIR = "reports/figures"
MODEL_DIR = "models"