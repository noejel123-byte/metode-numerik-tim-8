"""
TIM 8 - Analisis Kinerja Siklus Otto
Metode: regresi, interpolasi, integrasi numerik (aturan trapesium).
Jalankan dari folder utama: python src/analisis_otto.py
Dependensi: numpy, pandas, matplotlib
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameter asumsi model udara-standar
P1, T1 = 100.0, 300.0       # kPa, K
R, gamma = 0.287, 1.4       # kJ/(kg K), rasio kalor jenis
r, qin = 8.0, 800.0         # rasio kompresi, kJ/kg
cv = R/(gamma-1)

# Keadaan 1-2-3-4 siklus Otto ideal
v1 = R*T1/P1
v2 = v1/r
T2 = T1*r**(gamma-1)
P2 = P1*r**gamma
T3 = T2 + qin/cv
P3 = P2*T3/T2
T4 = T3/r**(gamma-1)
P4 = P3/r**gamma
qout = cv*(T4-T1)
w_analitik = qin-qout
eta_teori = 1-1/(r**(gamma-1))

# Data tiap proses: isentropik dan volume konstan
v12 = np.linspace(v1,v2,500); p12 = P1*(v1/v12)**gamma
v23 = np.full(100,v2); p23 = np.linspace(P2,P3,100)
v34 = np.linspace(v2,v1,500); p34 = P3*(v2/v34)**gamma
v41 = np.full(100,v1); p41 = np.linspace(P4,P1,100)

# METODE 1 - REGRESI: p = C*v^(-n) melalui regresi ln(p) terhadap ln(v)
coef_c = np.polyfit(np.log(v12),np.log(p12),1)
coef_e = np.polyfit(np.log(v34),np.log(p34),1)
n_c,C_c = -coef_c[0],np.exp(coef_c[1])
n_e,C_e = -coef_e[0],np.exp(coef_e[1])

# METODE 2 - INTERPOLASI: ambil 25 titik data lalu interpolasi linier
idx_c = np.unique(np.linspace(0,len(v12)-1,25).astype(int))
idx_e = np.unique(np.linspace(0,len(v34)-1,25).astype(int))
v_c_data,p_c_data = v12[idx_c],p12[idx_c]
v_e_data,p_e_data = v34[idx_e],p34[idx_e]
v_c_grid = np.linspace(v1,v2,200)
v_e_grid = np.linspace(v2,v1,200)
p_c_interp = np.interp(v_c_grid,v_c_data[::-1],p_c_data[::-1])
p_e_interp = np.interp(v_e_grid,v_e_data,p_e_data)

# METODE 3 - INTEGRASI: kerja batas W = integral(P dV), aturan trapesium
w_kompresi = np.trapezoid(p12,v12)
w_ekspansi = np.trapezoid(p34,v34)
w_numerik = w_kompresi+w_ekspansi
w_interpolasi = np.trapezoid(p_c_interp,v_c_grid)+np.trapezoid(p_e_interp,v_e_grid)

# Validasi terhadap neraca energi dan efisiensi ideal
galat_persen = abs(w_numerik-w_analitik)/abs(w_analitik)*100
eta_numerik = w_numerik/qin
states = pd.DataFrame({
 "Titik":[1,2,3,4],
 "Tekanan_kPa":[P1,P2,P3,P4],
 "Temperatur_K":[T1,T2,T3,T4],
 "Volume_spesifik_m3_kg":[v1,v2,v2,v1]
})
states.to_csv("data/keadaan_siklus.csv",index=False)

plt.figure(figsize=(8,5))
plt.plot(v12,p12,label="1-2 Kompresi")
plt.plot(v23,p23,label="2-3 Kalor masuk")
plt.plot(v34,p34,label="3-4 Ekspansi")
plt.plot(v41,p41,label="4-1 Kalor keluar")
for i,(v,p) in enumerate([(v1,P1),(v2,P2),(v2,P3),(v1,P4)],1):
 plt.scatter(v,p); plt.annotate(str(i),(v,p),xytext=(5,5),textcoords="offset points")
plt.xlabel("Volume spesifik (m³/kg)"); plt.ylabel("Tekanan (kPa)")
plt.title("Diagram P-V Siklus Otto Ideal"); plt.grid(True); plt.legend()
plt.tight_layout(); plt.savefig("hasil/diagram_PV.png",dpi=180); plt.show()

print("=== HASIL ANALISIS SIKLUS OTTO TIM 8 ===")
print(states.to_string(index=False))
print(f"Regresi kompresi: p = {C_c:.5f} * v^(-{n_c:.5f})")
print(f"Regresi ekspansi: p = {C_e:.5f} * v^(-{n_e:.5f})")
print(f"Kerja numerik trapesium : {w_numerik:.4f} kJ/kg")
print(f"Kerja interpolasi       : {w_interpolasi:.4f} kJ/kg")
print(f"Kerja analitik          : {w_analitik:.4f} kJ/kg")
print(f"Galat kerja             : {galat_persen:.6f} %")
print(f"Efisiensi numerik       : {eta_numerik*100:.4f} %")
print(f"Efisiensi Otto teoritis : {eta_teori*100:.4f} %")
