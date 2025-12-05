import cv2
import face_recognition
import numpy as np
import time
import requests
import json
import os
import pickle
import threading
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, send_from_directory
from gtts import gTTS

# ==================== НАСТРОЙКИ ====================
RTSP_URL = "rtsp://admin:password@192.168.1.100:554/stream"  # Замените на ваш RTSP URL из Trassir
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
DATABASE_PATH = "face_database.pkl"
UPLOAD_FOLDER = "detected_images"
NOTIFICATION_COOLDOWN = 30  # секунд между оповещениями для одного лица
# ===================================================

# Создаем папку для изображений
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Загрузка базы известных лиц
def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    
    if os.path.exists(DATABASE_PATH):
        with open(DATABASE_PATH, "rb") as f:
            data = pickle.load(f)
            known_face_encodings = data["encodings"]
            known_face_names = data["names"]
        print(f"Загружено {len(known_face_names)} известных лиц")
    else:
        print("База лиц не найдена, создается новая")
    
    return known_face_encodings, known_face_names

# Сохранение базы лиц
def save_face_database(encodings, names):
    with open(DATABASE_PATH, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)
    print(f"База сохранена: {len(names)} лиц")

# Добавление лица в базу
def add_face_to_database(image_path, name, known_encodings, known_names):
    try:
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        
        if face_encodings:
            known_encodings.append(face_encodings[0])
            known_names.append(name)
            save_face_database(known_encodings, known_names)
            print(f"✅ Лицо '{name}' добавлено в базу")
            return True
        else:
            print("❌ Не удалось найти лицо на изображении")
            return False
    except Exception as e:
        print(f"❌ Ошибка при добавлении лица: {e}")
        return False

# ==================== FLASK ВЕБ-СЕРВЕР ====================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
notifications = []
system_active = True

# HTML шаблон в виде строки
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Система распознавания лиц Trassir</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            color: white;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        .status-bar {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
        }
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 1.1em;
            font-weight: bold;
            color: white;
        }
        .status-dot {
            width: 15px;
            height: 15px;
            border-radius: 50%;
            background: #28a745;
            box-shadow: 0 0 10px #28a745;
        }
        .notifications-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 25px;
        }
        .notification-card {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .notification-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.3);
        }
        .card-header {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .card-name {
            font-size: 1.4em;
            font-weight: bold;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }
        .card-time {
            background: rgba(255,255,255,0.2);
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 0.9em;
            font-weight: 500;
        }
        .card-content {
            padding: 20px;
        }
        .card-image {
            width: 100%;
            height: 250px;
            object-fit: cover;
            border-radius: 8px;
            margin-bottom: 15px;
            border: 2px solid #e9ecef;
        }
        .card-audio {
            width: 100%;
            margin-top: 10px;
        }
        .controls {
            display: flex;
            gap: 15px;
            margin: 20px 0;
            flex-wrap: wrap;
            justify-content: center;
        }
        .btn {
            padding: 12px 25px;
            border: none;
            border-radius: 50px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .btn-primary {
            background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
            color: white;
        }
        .btn-success {
            background: linear-gradient(135deg, #00d2ff 0%, #0f62fe 100%);
            color: white;
        }
        .btn-danger {
            background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
            color: white;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        .btn:active {
            transform: translateY(0);
        }
        .stats {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 30px 0;
            flex-wrap: wrap;
        }
        .stat-item {
            background: rgba(255,255,255,0.9);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            min-width: 150px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
            margin: 5px 0;
        }
        .stat-label {
            color: #6c757d;
            font-size: 0.9em;
        }
        @media (max-width: 768px) {
            .notifications-container {
                grid-template-columns: 1fr;
            }
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📹 Система распознавания лиц Trassir</h1>
            <p style="font-size: 1.1em; opacity: 0.9;">Распознавание лиц в реальном времени с RTSP потока</p>
        </div>

        <div class="status-bar">
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span>Система активна</span>
            </div>
            <div class="status-indicator">
                <span>🕒 Последнее обновление: <span id="last-update">{{ last_update }}</span></span>
            </div>
        </div>

        <div class="stats">
            <div class="stat-item">
                <div class="stat-number">{{ total_detections }}</div>
                <div class="stat-label">Всего обнаружений</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{{ len(notifications) }}</div>
                <div class="stat-label">Активные оповещения</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{{ known_faces_count }}</div>
                <div class="stat-label">Известных лиц</div>
            </div>
        </div>

        <div class="controls">
            <button class="btn btn-primary" onclick="refreshPage()">
                <i>🔄</i> Обновить данные
            </button>
            <button class="btn btn-success" onclick="showAddFaceModal()">
                <i>👤</i> Добавить лицо
            </button>
            <button class="btn btn-danger" onclick="clearNotifications()">
                <i>🗑️</i> Очистить оповещения
            </button>
        </div>

        <div class="notifications-container" id="notifications-container">
            {% for notification in notifications %}
            <div class="notification-card" id="notif-{{ notification.id }}">
                <div class="card-header">
                    <div class="card-name">👤 {{ notification.name }}</div>
                    <div class="card-time">⏰ {{ notification.timestamp }}</div>
                </div>
                <div class="card-content">
                    <img src="{{ url_for('get_image', filename=notification.image_path) }}" 
                         alt="Обнаруженное лицо" class="card-image">
                    <div style="text-align: center; margin-top: 10px; color: #6c757d;">
                        📍 {{ notification.location }}
                    </div>
                    <audio controls class="card-audio">
                        <source src="{{ url_for('get_audio', filename=notification.voice_path) }}" type="audio/mpeg">
                        Ваш браузер не поддерживает аудио.
                    </audio>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>

    <div id="add-face-modal" style="display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.7); z-index: 1000; display: flex; align-items: center; justify-content: center;">
        <div style="background: white; padding: 30px; border-radius: 15px; max-width: 500px; width: 90%;">
            <h2 style="text-align: center; margin-bottom: 20px; color: #007bff;">➕ Добавить новое лицо</h2>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: bold;">Имя человека:</label>
                <input type="text" id="person-name" style="width: 100%; padding: 10px; border: 2px solid #ddd; border-radius: 8px; font-size: 1.1em;">
            </div>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: bold;">Путь к изображению:</label>
                <input type="text" id="image-path" style="width: 100%; padding: 10px; border: 2px solid #ddd; border-radius: 8px; font-size: 1.1em;" placeholder="/path/to/photo.jpg">
            </div>
            <div style="display: flex; gap: 15px; justify-content: center; margin-top: 20px;">
                <button class="btn btn-success" onclick="addFaceToDatabase()" style="flex: 1;">
                    <i>✅</i> Добавить
                </button>
                <button class="btn btn-danger" onclick="closeModal()" style="flex: 1;">
                    <i>❌</i> Отмена
                </button>
            </div>
        </div>
    </div>

    <script>
        // Автообновление каждые 5 секунд
        setInterval(updateNotifications, 5000);
        updateNotifications();
        
        function updateNotifications() {
            fetch('/api/notifications')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('notifications-container').innerHTML = data.html;
                    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                    document.querySelector('.stat-number:first-child').textContent = data.total_detections;
                    document.querySelectorAll('.stat-number')[1].textContent = data.active_notifications;
                    document.querySelectorAll('.stat-number')[2].textContent = data.known_faces_count;
                })
                .catch(error => console.error('Ошибка обновления:', error));
        }
        
        function refreshPage() {
            location.reload();
        }
        
        function showAddFaceModal() {
            document.getElementById('add-face-modal').style.display = 'flex';
        }
        
        function closeModal() {
            document.getElementById('add-face-modal').style.display = 'none';
        }
        
        function addFaceToDatabase() {
            const name = document.getElementById('person-name').value;
            const imagePath = document.getElementById('image-path').value;
            
            if (!name || !imagePath) {
                alert('Пожалуйста, заполните все поля!');
                return;
            }
            
            fetch('/api/add_face', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    name: name,
                    image_path: imagePath
                })
            })
            .then(response => response.json())
            .then(data => {
                alert(data.message);
                if (data.success) {
                    closeModal();
                    updateNotifications();
                }
            })
            .catch(error => {
                alert('Ошибка при добавлении лица: ' + error);
            });
        }
        
        function clearNotifications() {
            if (confirm('Вы уверены, что хотите очистить все оповещения?')) {
                fetch('/api/clear_notifications', {
                    method: 'POST'
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateNotifications();
                        alert('Оповещения успешно очищены!');
                    }
                });
            }
        }
        
        // Закрытие модального окна при клике вне его
        window.addEventListener('click', function(e) {
            if (e.target.id === 'add-face-modal') {
                closeModal();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    known_encodings, known_names = load_known_faces()
    last_update = datetime.now().strftime("%H:%M:%S")
    total_detections = len(notifications)
    
    return render_template_string(HTML_TEMPLATE, 
                                 notifications=notifications,
                                 last_update=last_update,
                                 total_detections=total_detections,
                                 known_faces_count=len(known_names))

@app.route('/api/notifications')
def api_notifications():
    known_encodings, known_names = load_known_faces()
    
    # Генерируем HTML для оповещений
    notifications_html = ""
    for notification in notifications:
        notifications_html += f"""
        <div class="notification-card" id="notif-{notification['id']}">
            <div class="card-header">
                <div class="card-name">👤 {notification['name']}</div>
                <div class="card-time">⏰ {notification['timestamp']}</div>
            </div>
            <div class="card-content">
                <img src="/images/{notification['image_path']}" class="card-image">
                <div style="text-align: center; margin-top: 10px; color: #6c757d;">
                    📍 Камера 1
                </div>
                <audio controls class="card-audio">
                    <source src="/audio/{notification['voice_path']}" type="audio/mpeg">
                </audio>
            </div>
        </div>
        """
    
    return jsonify({
        'html': notifications_html,
        'total_detections': len(notifications),
        'active_notifications': len(notifications),
        'known_faces_count': len(known_names)
    })

@app.route('/api/add_face', methods=['POST'])
def api_add_face():
    data = request.json
    name = data.get('name')
    image_path = data.get('image_path')
    
    if not name or not image_path:
        return jsonify({
            'success': False,
            'message': 'Необходимо указать имя и путь к изображению'
        })
    
    known_encodings, known_names = load_known_faces()
    
    if add_face_to_database(image_path, name, known_encodings, known_names):
        return jsonify({
            'success': True,
            'message': f'Лицо "{name}" успешно добавлено в базу!'
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Не удалось добавить лицо. Проверьте путь к изображению и наличие лица на фото.'
        })

@app.route('/api/clear_notifications', methods=['POST'])
def api_clear_notifications():
    global notifications
    notifications = []
    return jsonify({'success': True})

@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/audio/<filename>')
def get_audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ==================== СИСТЕМА РАСПОЗНАВАНИЯ ЛИЦ ====================
class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings, self.known_face_names = load_known_faces()
        self.process_this_frame = True
        self.last_notification_time = {}
        self.notification_id_counter = 1
        self.frame_count = 0
        self.total_detections = 0
        
    def generate_voice_notification(self, name):
        """Генерация голосового оповещения"""
        try:
            text = f"Внимание! Обнаружено лицо: {name}"
            tts = gTTS(text=text, lang='ru')
            filename = f"voice_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            tts.save(filepath)
            return filename
        except Exception as e:
            print(f"❌ Ошибка генерации голоса: {e}")
            return None
    
    def send_notification(self, name, timestamp, frame, location="Камера 1"):
        """Отправка оповещения и сохранение данных"""
        try:
            # Сохранение кадра
            filename = f"detected_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            cv2.imwrite(filepath, frame)
            
            # Генерация голосового оповещения
            voice_file = self.generate_voice_notification(name)
            
            # Создание записи об оповещении
            notification = {
                'id': self.notification_id_counter,
                'name': name,
                'timestamp': timestamp,
                'image_path': filename,
                'voice_path': voice_file,
                'location': location
            }
            
            notifications.insert(0, notification)  # Добавляем в начало списка
            self.notification_id_counter += 1
            self.total_detections += 1
            
            print(f"🔔 Оповещение создано для {name}")
            return True
        except Exception as e:
            print(f"❌ Ошибка при создании оповещения: {e}")
            return False
    
    def process_frame(self, frame):
        """Обработка одного кадра"""
        self.frame_count += 1
        
        # Обрабатываем каждый 2-й кадр для повышения производительности
        if self.frame_count % 2 != 0:
            return frame
        
        # Изменение размера для ускорения обработки
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Обнаружение лиц
        face_locations = face_recognition.face_locations(rgb_small_frame)
        
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"
                
                if matches and self.known_face_encodings:
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                        name = self.known_face_names[best_match_index]
                
                face_names.append(name)
                
                # Если найдено известное лицо и прошло достаточно времени с последнего оповещения
                current_time = time.time()
                if name != "Unknown" and name in self.known_face_names:
                    if (name not in self.last_notification_time or 
                        current_time - self.last_notification_time[name] > NOTIFICATION_COOLDOWN):
                        
                        self.last_notification_time[name] = current_time
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Отправка оповещения в отдельном потоке
                        threading.Thread(
                            target=self.send_notification, 
                            args=(name, timestamp, frame.copy()),
                            daemon=True
                        ).start()
        
            # Рисование прямоугольников и имен
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Масштабируем обратно
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
                
                # Рисуем прямоугольник
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Подпись с именем
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Основной цикл обработки видео"""
        print("📹 Запуск системы распознавания лиц...")
        print(f"📡 RTSP URL: {RTSP_URL}")
        
        cap = cv2.VideoCapture(RTSP_URL)
        
        if not cap.isOpened():
            print("❌ Ошибка подключения к RTSP потоку")
            print("Проверьте:")
            print(f"1. Правильность RTSP URL: {RTSP_URL}")
            print("2. Доступность камеры в сети")
            print("3. Правильность логина и пароля")
            return
        
        print("✅ Подключение установлено")
        print(f"👥 Загружено лиц в базе: {len(self.known_face_names)}")
        print("Нажмите 'q' для выхода")
        
        while system_active:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Ошибка чтения кадра, попытка переподключения...")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(RTSP_URL)
                continue
            
            # Обработка кадра
            processed_frame = self.process_frame(frame)
            
            # Отображение кадра (только при локальном запуске)
            try:
                cv2.imshow('Face Recognition System', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                pass  # Игнорируем ошибки отображения при запуске на сервере без GUI
            
            # Ограничение FPS для снижения нагрузки
            time.sleep(0.05)
        
        cap.release()
        cv2.destroyAllWindows()
        print("⏹️ Система остановлена")

# ==================== ОСНОВНОЙ КОД ====================
def run_web_server():
    """Запуск Flask веб-сервера в отдельном потоке"""
    print(f"🌐 Запуск веб-сервера на http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)

def main():
    """Основная функция запуска системы"""
    global system_active
    
    try:
        # Инициализация системы распознавания
        face_system = FaceRecognitionSystem()
        
        # Запуск веб-сервера в отдельном потоке
        web_thread = threading.Thread(target=run_web_server, daemon=True)
        web_thread.start()
        
        # Запуск системы распознавания лиц
        face_system.run()
        
    except KeyboardInterrupt:
        print("\n⏹️ Получен сигнал остановки...")
        system_active = False
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        system_active = False
    
    print("✅ Система успешно завершена")

if __name__ == "__main__":
    # Проверка установки необходимых библиотек
    required_packages = ['cv2', 'face_recognition', 'numpy', 'flask', 'gtts']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Отсутствуют необходимые библиотеки:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nУстановите их командой:")
        print("pip install opencv-python face_recognition numpy flask gTTS requests")
        exit(1)
    
    # Запуск системы
    print("=" * 60)
    print("🚀 СИСТЕМА РАСПОЗНАВАНИЯ ЛИЦ TRASSIR")
    print("=" * 60)
    main()
