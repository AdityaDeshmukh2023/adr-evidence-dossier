"""Optional research prediction package; core screening does not import ML dependencies."""

TASK = 'documented_interaction_severity'
CLASSES = ('high', 'moderate', 'low')
MODEL_NAMES = ('fingerprint', 'graphsage', 'fusion', 'role_fusion')

