// Webcam Setup
const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const captureBtn = document.getElementById('captureBtn');
const resultDiv = document.getElementById('result');
const previewImg = document.getElementById('preview');

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => video.srcObject = stream)
    .catch(err => console.error("Webcam error: ", err));

// Capture from webcam
captureBtn.addEventListener('click', () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('file', blob, 'webcam.jpg');
        sendToServer(formData);
    }, 'image/jpeg');
});

// Handle file upload
document.getElementById('uploadForm').addEventListener('submit', (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    sendToServer(formData);
});

function sendToServer(formData) {
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultDiv.textContent = `Error: ${data.error}`;
        } else {
            resultDiv.textContent = `Prediction: ${data.prediction} (Confidence: ${data.confidence})`;
            if (data.image_path) {
                previewImg.src = data.image_path;
                previewImg.style.display = 'block';
            }
        }
    })
    .catch(error => {
        resultDiv.textContent = `Error: ${error.message}`;
    });
}