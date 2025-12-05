import cv2
import face_recognition
import threading
import time
from flask import Flask, render_template_string, jsonify
from gtts import gTTS
import os

# Конфигурация
rtsp_url = 'rtsp://your_username:your_password@your_ip:your_port/your_stream'  # Замени на свой RTSP URL из Trassir
known_faces_dir = 'known_faces'  # Папка с изображениями известных лиц
tolerance = 0.6  # Порог совпадения (меньше — строже)
alert_message = "Обнаружено совпадение лица!"  # Текст оповещения
alerts = []  # Список оповещений для веб-интерфейса (лог событий)

# Создаём папку static, если нет
if not os.path.exists('static'):
    os.makedirs('static')

# Загрузка известных лиц
known_face_encodings = []
known_face_names = []

for filename in os.listdir(known_faces_dir):
    if filename.endswith(('.jpg', '.png')):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])  # Имя без расширения

print(f"Загружено {len(known_face_encodings)} известных лиц.")

# Функция обработки видео
def process_video():
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть RTSP поток.")
        return

    last_alert_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения кадра. Переподключение...")
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            time.sleep(5)
            continue

        # Уменьшаем размер кадра для скорости
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Детекция лиц
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
            name = "Неизвестный"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

                # Проверяем, чтобы не спамить оповещениями (раз в 10 сек)
                if time.time() - last_alert_time > 10:
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    alert = f"[{timestamp}] Совпадение: {name}"
                    alerts.append(alert)
                    print(alert)

                    # Генерация голосового оповещения для веб
                    tts = gTTS(text=alert_message + f" {name}", lang='ru')
                    audio_file = "static/alert.mp3"
                    tts.save(audio_file)

                    last_alert_time = time.time()

        time.sleep(0.1)  # Пауза для снижения нагрузки

    cap.release()

# Flask веб-сервер
app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Лог событий совпадений лиц</title>
        <script>
            var audio = new Audio('/static/alert.mp3');
            var lastAlertsLength = 0;

            function refreshAlerts() {
                fetch('/alerts')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('alerts').innerHTML = data.alerts.map(a => '<li>' + a + '</li>').join('');
                        if (data.alerts.length > lastAlertsLength) {
                            audio.load();  // Перезагружаем аудио, если файл обновлён
                            audio.play().catch(function(error) {
                                console.log('Автовоспроизведение заблокировано: ' + error);
                            });
                            lastAlertsLength = data.alerts.length;
                        }
                    });
            }
            setInterval(refreshAlerts, 2000);  // Обновление каждые 2 сек
            refreshAlerts();  // Начальный вызов
        </script>
    </head>
    <body>
        <h1>Лог событий совпадений лиц</h1>
        <ul id="alerts"></ul>
    </body>
    </html>
    ''')

@app.route('/alerts')
def get_alerts():
    return jsonify({'alerts': alerts[-50:]})  # Последние 50 событий для лога (чтобы не перегружать)

if __name__ == '__main__':
    # Запуск обработки видео в отдельном потоке
    video_thread = threading.Thread(target=process_video, daemon=True)
    video_thread.start()

    # Запуск Flask
    app.run(host='0.0.0.0', port=5000, debug=True)
