import pandas as pd  # untuk manipulasi data
import numpy as np  # untuk operasi numerik
import tkinter as tk  # untuk antarmuka pengguna grafis
from tkinter import ttk, filedialog, messagebox  # untuk widget, dialog file, dan pesan di tkinter
import warnings  # untuk menangani peringatan
from statsmodels.tsa.holtwinters import SimpleExpSmoothing  # untuk model peramalan sederhana
import matplotlib.pyplot as plt  # untuk plotting grafik
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # untuk integrasi matplotlib dengan tkinter

class SalesPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sales Prediction")
        self.root.geometry("1280x720")

        self.data = None  # untuk menyimpan data yang diimpor
        self.predictions = None  # untuk menyimpan prediksi penjualan
        self.mse_values = None  # untuk menyimpan nilai MSE

        self.create_widgets()  # membuat widget GUI
        self.load_default_file()  # memuat file default jika ada

    def create_widgets(self):
        # Label Visualisasi
        self.label = tk.Label(self.root, text="Visualisasi", font=("Arial", 24))
        self.label.pack(pady=10)

        # Frame untuk ComboBox dan Tabel di sebelah kiri
        self.left_frame = tk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # ComboBox untuk memilih item
        self.combobox_label = tk.Label(self.left_frame, text="Pilih Barang", font=("Arial", 14))
        self.combobox_label.pack(pady=5)
        self.combobox = ttk.Combobox(self.left_frame, state="readonly")
        self.combobox.pack(pady=5)
        self.combobox.bind("<<ComboboxSelected>>", self.update_predictions)

        # Treeview untuk menampilkan prediksi dan MSE
        self.tree = ttk.Treeview(self.left_frame, columns=("alpha", "prediction", "mse"), show="headings")
        self.tree.heading("alpha", text="Alpha")
        self.tree.heading("prediction", text="Prediction")
        self.tree.heading("mse", text="MSE")
        self.tree.pack(pady=10)

        # Frame untuk Grafik di sebelah kanan
        self.right_frame = tk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Figure untuk plotting grafik
        self.figure = plt.Figure(figsize=(5, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tombol untuk memilih file
        self.file_button = tk.Button(self.root, text="Pilih File", command=self.load_file)
        self.file_button.pack(pady=20)

    def load_default_file(self):
        try:
            self.data = pd.read_csv("dataset.csv")  # memuat data dari file CSV default
            self.process_data()  # memproses data yang diimpor
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat file default: {e}")

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])  # membuka dialog file untuk memilih file CSV
        if file_path:
            try:
                self.data = pd.read_csv(file_path)  # memuat data dari file CSV yang dipilih
                self.process_data()  # memproses data yang diimpor
            except Exception as e:
                messagebox.showerror("Error", f"Gagal memuat file: {e}")
        else:
            messagebox.showerror("Error", "File tidak valid")

    def process_data(self):
        self.data['tanggal'] = pd.to_datetime(self.data['tahun'].astype(str) + '-' + self.data['bulan'].astype(str) + '-01')  # menggabungkan tahun dan bulan menjadi kolom tanggal
        self.data.set_index('tanggal', inplace=True)  # mengatur kolom tanggal sebagai indeks
        self.update_combobox()  # memperbarui pilihan di combobox

    def update_combobox(self):
        items = self.data['barang'].drop_duplicates().tolist()  # mendapatkan daftar barang unik
        self.combobox['values'] = items  # mengatur nilai combobox
        if items:
            self.combobox.current(0)  # memilih item pertama secara default
            self.update_predictions()  # memperbarui prediksi berdasarkan item yang dipilih

    def update_predictions(self, event=None):
        selected_item = self.combobox.get()  # mendapatkan item yang dipilih dari combobox
        if selected_item:
            item_data = self.data[self.data['barang'] == selected_item]['penjualan'].resample('M').sum()  # meresampling data penjualan per bulan
            self.predictions = {}
            self.mse_values = {}
            alphas = np.arange(0.1, 1.0, 0.1)  # menentukan rentang nilai alpha
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for alpha in alphas:
                    model = SimpleExpSmoothing(item_data).fit(smoothing_level=alpha)  # membuat model SimpleExponentialSmoothing
                    pred = model.fittedvalues  # mendapatkan nilai prediksi yang sesuai dengan data aktual
                    pred = pd.concat([pred, model.forecast(1)])  # menambahkan satu prediksi ke depan
                    self.predictions[f'{alpha:.1f}'] = round(pred.iloc[-1], 1)  # menyimpan prediksi dengan pembulatan satu desimal

                    # Menghitung MSE (Mean Squared Error)
                    mse = ((pred[:-1] - item_data) ** 2).mean()
                    self.mse_values[f'{alpha:.1f}'] = mse

            # Memilih prediksi dengan MSE terendah
            best_alpha = min(self.mse_values, key=self.mse_values.get)
            best_prediction = self.predictions[best_alpha]

            self.update_table()  # memperbarui tabel dengan prediksi baru
            self.update_graph(item_data, best_alpha, best_prediction)  # memperbarui grafik dengan prediksi baru

    def update_table(self):
        for row in self.tree.get_children():
            self.tree.delete(row)  # menghapus semua baris dalam tabel
        for alpha, prediction in self.predictions.items():
            mse = round(self.mse_values[alpha], 1)  # Membatasi MSE menjadi 1 angka setelah koma
            self.tree.insert("", "end", values=(alpha, f'{prediction:.1f}', f'{mse:.1f}'))  # menambahkan baris baru ke tabel

    def update_graph(self, item_data, best_alpha, best_prediction):
        self.ax.clear()  # membersihkan grafik sebelumnya
        self.ax.plot(item_data, label="Penjualan Aktual")  # plotting data penjualan aktual
        last_date = item_data.index[-1]
        forecast_index = last_date + pd.DateOffset(months=1)  # menentukan indeks untuk prediksi
        self.ax.plot([last_date, forecast_index], [item_data[-1], best_prediction], linestyle='--', label=f'Alpha {best_alpha}')  # plotting garis prediksi
        self.ax.scatter(forecast_index, best_prediction, color='red')  # menambahkan titik prediksi
        self.ax.legend()  # menambahkan legenda
        self.ax.set_title("Prediksi Penjualan Bulan Berikutnya (dengan MSE terendah)")  # mengatur judul grafik
        self.ax.set_xlabel("Tanggal")  # mengatur label sumbu x
        self.ax.set_ylabel("Penjualan")  # mengatur label sumbu y
        self.canvas.draw()  # menggambar ulang canvas

if __name__ == "__main__":
    root = tk.Tk()  # membuat root window tkinter
    app = SalesPredictionApp(root)  # membuat instance dari SalesPredictionApp
    root.mainloop()  # menjalankan aplikasi
