-- phpMyAdmin SQL Dump
-- version 5.1.0
-- https://www.phpmyadmin.net/
--
-- Host: localhost
-- Waktu pembuatan: 21 Nov 2025 pada 04.34
-- Versi server: 10.4.19-MariaDB
-- Versi PHP: 8.0.6

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `pneumonia_db`
--

-- --------------------------------------------------------

--
-- Struktur dari tabel `patient`
--

CREATE TABLE `patient` (
  `id` int(11) NOT NULL,
  `nama` varchar(100) DEFAULT NULL,
  `umur` int(11) DEFAULT NULL,
  `alamat` varchar(200) DEFAULT NULL,
  `jenis_kelamin` varchar(10) DEFAULT NULL,
  `file_path` varchar(200) DEFAULT NULL,
  `predicted_label` varchar(100) DEFAULT NULL,
  `confidence` float DEFAULT NULL,
  `severity` varchar(50) DEFAULT NULL,
  `uploaded_at` datetime DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data untuk tabel `patient`
--

INSERT INTO `patient` (`id`, `nama`, `umur`, `alamat`, `jenis_kelamin`, `file_path`, `predicted_label`, `confidence`, `severity`, `uploaded_at`) VALUES
(152, 'muhamad zammah syari', 45, 'temanggunf', 'Perempuan', 'static/uploads/IM-0049-0001.jpeg', 'NORMAL', 0.636364, 'Normal', '2025-11-02 21:22:49'),
(153, 'muhamad zammah syari', 23, 'temanggung', 'Laki-laki', 'static/uploads/IM-0006-0001.jpeg', 'PNEUMONIA', 0.909091, 'Berat', '2025-11-03 07:37:57'),
(154, 'muhamad zammah syari', 23, 'temanggung', 'Laki-laki', 'static/uploads/IM-0001-0001.jpeg', 'NORMAL', 0.909091, 'Normal', '2025-11-03 07:38:28'),
(155, 'muhamad zammah syari', 23, 'temanggung', 'Laki-laki', 'static/uploads/IM-0001-0001.jpeg', 'NORMAL', 0.909091, 'Normal', '2025-11-03 07:52:21'),
(156, 'muhamad zammah syari', 23, 'temanggung', 'Laki-laki', 'static/uploads/IM-0001-0001.jpeg', 'NORMAL', 0.909091, 'Normal', '2025-11-03 07:54:39'),
(157, 'muhamad zammah syari', 23, 'temanggung', 'Laki-laki', 'static/uploads/IM-0001-0001.jpeg', 'NORMAL', 0.909091, 'Normal', '2025-11-03 07:55:30'),
(158, 'muhamad zammah syari', 23, 'temanggung', 'Laki-laki', 'static/uploads/IM-0001-0001.jpeg', 'NORMAL', 0.909091, 'Normal', '2025-11-03 07:55:42'),
(159, 'muhamad zammah syari', 23, 'temanggung', 'Laki-laki', 'static/uploads/IM-0001-0001.jpeg', 'NORMAL', 0.909091, 'Normal', '2025-11-03 07:56:02'),
(160, 'muhamad zammah syari', 23, 'temanggung', 'Laki-laki', 'static/uploads/IM-0001-0001.jpeg', 'NORMAL', 0.909091, 'Normal', '2025-11-03 07:56:16'),
(161, 'muhamad zammah', 21, 'temanggung', 'Laki-laki', 'static/uploads/person1_virus_6.jpeg', 'PNEUMONIA', 1, 'Berat', '2025-11-03 08:31:40'),
(162, 'muhamad zammah', 23, 'magelang', 'Perempuan', 'static/uploads/IM-0003-0001.jpeg', 'NORMAL', 1, 'Normal', '2025-11-03 08:33:10'),
(163, 'muhamad zam', 23, 'temanggung', 'Laki-laki', 'static/uploads/person5_bacteria_17.jpeg', 'PNEUMONIA', 1, 'Berat', '2025-11-03 08:34:33'),
(164, 'zamah', 34, 'temanggug', 'Perempuan', 'static/uploads/person5_bacteria_16.jpeg', 'PNEUMONIA', 1, 'Berat', '2025-11-03 08:44:31'),
(165, 'zam', 34, 'frf', 'Laki-laki', 'static/uploads/person2_bacteria_3.jpeg', 'PNEUMONIA', 1, 'Berat', '2025-11-03 08:49:07'),
(166, 'zamah', 23, 'temanggung', 'Perempuan', 'static/uploads/IM-0001-0001.jpeg', 'NORMAL', 0.909091, 'Normal', '2025-11-20 09:57:17'),
(167, 'Pasien API', 0, 'Tidak diketahui', '-', 'static/uploads/scaled_1000000021.jpg', 'NORMAL', 1, 'Normal', '2025-11-20 10:00:04'),
(168, 'Pasien API', 0, 'Tidak diketahui', '-', 'static/uploads/scaled_1000000021.jpg', 'NORMAL', 1, 'Normal', '2025-11-20 10:00:35');

-- --------------------------------------------------------

--
-- Struktur dari tabel `patients`
--

CREATE TABLE `patients` (
  `id` int(11) NOT NULL,
  `nama` varchar(100) DEFAULT NULL,
  `umur` int(11) DEFAULT NULL,
  `alamat` varchar(255) DEFAULT NULL,
  `jenis_kelamin` varchar(10) DEFAULT NULL,
  `file_path` varchar(255) DEFAULT NULL,
  `predicted_label` varchar(100) DEFAULT NULL,
  `confidence` float DEFAULT NULL,
  `severity` varchar(50) DEFAULT NULL,
  `uploaded_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Indexes for dumped tables
--

--
-- Indeks untuk tabel `patient`
--
ALTER TABLE `patient`
  ADD PRIMARY KEY (`id`);

--
-- Indeks untuk tabel `patients`
--
ALTER TABLE `patients`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT untuk tabel yang dibuang
--

--
-- AUTO_INCREMENT untuk tabel `patient`
--
ALTER TABLE `patient`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=169;

--
-- AUTO_INCREMENT untuk tabel `patients`
--
ALTER TABLE `patients`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
