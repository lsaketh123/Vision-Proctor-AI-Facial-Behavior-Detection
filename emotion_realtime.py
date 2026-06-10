# emotion_realtime.py
from deepface import DeepFace
import cv2
import pandas as pd
from datetime import datetime

cap = cv2.VideoCapture(0)
log_data = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    emotion = result[0]['dominant_emotion']
    score = result[0]['emotion'][emotion]

    # Add log
    log_data.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'emotion': emotion,
        'score': round(score, 2)
    })

    cv2.putText(frame, f"{emotion} ({score:.2f})", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save logs to CSV
pd.DataFrame(log_data).to_csv('emotion_log.csv', index=False)
print("✅ Log saved to emotion_log.csv")
