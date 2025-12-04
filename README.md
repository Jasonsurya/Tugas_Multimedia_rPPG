Laporan
-----
Pada tugas video processing ini tujuan utamanya adalah mendeteksi BPM dari video. Untuk melakukan hal tersebut, sistem harus bisa menangkap perubahan warna halus pada kulit, karena perubahan warna ini berkaitan dengan aliran darah yang dapat digunakan untuk memperkirakan detak jantung seseorang.
Pada kode asli di dalam video, proses pendeteksian wajah dilakukan dengan library dlib menggunakan model standar get_frontal_face_detector, kemudian landmark wajah diambil menggunakan shape_predictor_68_face_landmarks.dat. Dari landmark tersebut diambil area wajah sebagai ROI dan dilakukan ekstraksi sinyal RGB untuk dimasukkan ke metode POS. Setelah itu digunakan bandpass filter dengan rentang 0.8–4.0 Hz agar sinyal sesuai dengan rentang BPM manusia.

Perbedaan
------
Pada kode baru yang saya buat, terdapat beberapa perubahan dan perbaikan dibandingkan yang ada di dalam video. Pertama, saya tidak lagi menggunakan dlib, tetapi beralih ke MediaPipe FaceMesh karena landmark yang dihasilkan jauh lebih detail dan stabil. Kedua, saya mempersempit ROI hanya pada area pipi, karena bagian pipi memberikan sinyal yang lebih konsisten. Ketiga, saya menambahkan visualisasi sinyal secara realtime, baik sinyal RGB maupun sinyal hasil POS yang sudah difilter. Hal ini membantu untuk melihat apakah sinyal yang diambil sudah benar dan stabil.
Dengan perubahan ini, proses deteksi BPM menjadi lebih akurat dan hasilnya lebih stabil dibandingkan kode yang ada di video.
