# opensim-user

Package Python per creare utenti antropometrici OpenSim a partire dal modello full-body di Rajagopal-Lai-Uhlrich e dai riferimenti antropometrici ANSUR II. Il risultato è un oggetto `OpensimUser` con il modello OpenSim già caricato, scalato e pronto per analisi biomeccaniche, simulazioni e manipolazione della postura.

## Contenuto del progetto

```text
assets/
	ansur_ref.csv                  # riferimenti antropometrici ANSUR II
	rajagopalaiulrich2023.osim     # modello OpenSim base
	meshes/
		*.vtp                       # mesh per il rendering OpenSim
src/opensim_user/
	data.py                        # caricamento ANSUR e percentili
	mapping.py                     # mappa ANSUR -> corpi OpenSim
	model.py                       # facade e scaling OpenSim
	user.py                        # API pubblica OpensimUser
tests/                           # test statistici e integrazione OpenSim
```

Il modello referenzia 81 mesh VTP, tutte incluse nella cartella `assets/meshes/`. Sono presenti anche quattro alias aggiuntivi per femori e tibie.

## Installazione

Il package richiede Python 3.10 o superiore, NumPy e Pandas. I binding Python di OpenSim sono dipendenze native: su Windows è consigliato installarli in un ambiente Conda compatibile con la versione OpenSim utilizzata. 

Se l’ambiente supporta l’installazione del binding tramite pip, è possibile
usare anche l’extra opzionale:

```powershell
python -m pip install -e ".[test,opensim]"
```

L’importazione di `OpensimUser` non carica OpenSim immediatamente. Il binding è richiesto quando viene creata un’istanza della classe; in caso di ambiente non configurato viene sollevato un errore esplicito.

## Creare un utente

### Percentile predefinito

Passando solo il sesso, viene usato il 50° percentile per tutte le misure, statura inclusa:

```python
from opensim_user import OpensimUser

male = OpensimUser("M")
female = OpensimUser("F")

print(male.gender)       # "M"
print(male.percentile)   # 50.0
print(male.height)       # statura risolta in centimetri
```

`gender` accetta solo `"M"` e `"F"`.

### Percentile esplicito

```python
large_male = OpensimUser("M", percentile=75)

print(large_male.percentile)  # 75.0
print(large_male.height)      # statura al 75° percentile maschile
```

Il percentile deve appartenere all’intervallo inclusivo `[0.1, 99.9]`. Il calcolo usa direttamente `numpy.percentile` per ogni misura numerica ANSUR: non viene scelto un singolo soggetto e non viene effettuata interpolazione tra misure antropometriche.

### Altezza esplicita

L’altezza è espressa in centimetri. Il package calcola il percentile empirico della statura richiesta nel gruppo ANSUR del sesso indicato e applica quel percentile a tutte le misure:

```python
user = OpensimUser("M", height=175.0)

print(user.height)       # statura ANSUR risolta in centimetri
print(user.percentile)   # percentile empirico corrispondente a 175 cm
```

Quando `height` è presente, determina il percentile effettivo e ha precedenza sul valore passato in `percentile`. L’altezza deve essere positiva e compresa nel range osservato dal dataset ANSUR.

## Scaling antropometrico

Durante la costruzione di `OpensimUser` vengono eseguiti questi passaggi:

1. caricamento e normalizzazione del CSV ANSUR;
2. filtraggio per sesso;
3. calcolo del percentile comune per tutte le misure;
4. confronto con il riferimento del 50° percentile dello stesso sesso;
5. costruzione dei fattori `(x, y, z)` per i corpi OpenSim;
6. applicazione tramite `OpenSim.Model.scale` e `ScaleSet`.

La scalatura nativa OpenSim aggiorna in modo coordinato corpi, geometrie, frame articolari, marker e percorsi muscolari. La mappa usa direttamente le misure ANSUR disponibili per bacino, tronco, femori, tibie, piedi, braccia, avambracci e mani. Per assi o segmenti senza una misura ANSUR sufficientemente diretta viene usato il rapporto di statura come fallback deterministico.

Il file `.osim` originale e il CSV non vengono modificati. Ogni istanza possiede un modello OpenSim indipendente:

```python
user_50 = OpensimUser("M", percentile=50)
user_75 = OpensimUser("M", percentile=75)

assert user_50.model is not user_75.model
```

## Accesso al modello OpenSim

Le collezioni principali sono disponibili come proprietà:

```python
print(user.model)        # opensim.Model
print(user.state)        # opensim.State
print(user.bodies)       # BodySet
print(user.joints)       # JointSet
print(user.muscles)      # MuscleSet
print(user.markers)      # MarkerSet
print(user.coordinates)  # CoordinateSet
```

Per accedere a un elemento specifico si possono usare i metodi nominati:

```python
pelvis = user.body("pelvis")
hip = user.joint("hip_r")
gluteus = user.muscle("glut_max1_r")
marker = user.marker("RASI")
coordinate = user.coordinate("hip_flexion_r")
```

Il modello base contiene 22 corpi, 22 giunti, 80 muscoli, 66 marker e 39 coordinate.

## Modificare la postura

Gli angoli dell’API pubblica sono sempre espressi in gradi. La conversione in radianti viene effettuata internamente prima di chiamare OpenSim.

```python
user.set_right_hip_flexionextension(25.0)
user.set_left_hip_adduction(10.0)
user.set_right_knee_flexionextension(40.0)
user.set_left_ankle_flexiondorsiflexion(5.0)
user.set_right_shoulder_flexion(30.0)
user.set_left_elbow_flexion(90.0)
user.set_right_wrist_deviation(12.0)
user.set_lumbar_extension(8.0)

print(user.coordinate_degrees("hip_flexion_r"))  # 25.0
```

I setter che agiscono su una coordinata con lato (es. anca, ginocchio, caviglia, spalla, ecc.) sono disponibili in coppia `set_left_*`/`set_right_*`. Sono disponibili per:

- flessione/estensione ed adduzione/abduzione e rotazione dell’anca;
- flessione/estensione del ginocchio;
- flessione/dorsiflessione della caviglia;
- inversione subtalare e flessione MTP;
- flessione, adduzione e rotazione della spalla;
- flessione del gomito;
- flessione e deviazione del polso;
- pronazione/supinazione dell’avambraccio.

I setter lombari non hanno lato e restano invariati: estensione, inclinazione laterale e rotazione lombare.

È inoltre disponibile il setter generico:

```python
user.set_coordinate_degrees("arm_rot_r", 15.0)
angle = user.coordinate_degrees("arm_rot_r")
```

Le coordinate bloccate dal modello, come le coordinate subtalare e MTP di questa versione, non possono essere modificate: il relativo setter solleva `ValueError` invece di ignorare silenziosamente il valore.

## Riferimenti ANSUR senza OpenSim

Per usare soltanto il caricamento dati e i percentili:

```python
from opensim_user import load_ansur, resolve_reference

data = load_ansur("assets/ansur_ref.csv")
print(data.shape)

reference = resolve_reference(
		gender="F",
		percentile=75.0,
		dataset="assets/ansur_ref.csv",
)
print(reference.height_cm)
print(reference.values["footlength"])
```

Le misure lineari ANSUR restano nelle unità sorgente, prevalentemente millimetri. `stature_m` è in metri; `height_cm` è una proprietà di comodo in centimetri.

## Rendering

Le mesh si trovano in `assets/meshes/`; per visualizzare il modello occorre indicare quella cartella come percorso di ricerca della geometria:

```python
from opensim_user import OpensimViewer

user = OpensimUser("M", model_path="assets/rajagopalaiulrich2023.osim")
viewer = OpensimViewer(user, geometry_path="assets/meshes")
viewer.show()
```

Il file esportato con `user.export(...)` contiene il modello scalato; le mesh originali restano nella cartella `assets/meshes/`.

## Test

Per eseguire i test statistici e quelli di integrazione OpenSim:

```powershell
python -m pytest -q
```

I test di integrazione vengono saltati automaticamente solo se il modulo `opensim` non è importabile. Con i binding disponibili verificano caricamento, scaling, componenti del modello, indipendenza delle istanze, postura in gradi e gestione delle coordinate bloccate.

## Limiti e note

- Lo scaling è completo a livello di pipeline OpenSim, ma la qualità antropometrica dipende dalla corrispondenza tra misura ANSUR e segmento.
- Le misure senza corrispondenza diretta usano il rapporto di statura come fallback esplicito.
- L’altezza richiesta viene trasformata nel percentile ANSUR equivalente; per questo `user.height` può differire leggermente dall’input originale.
- Sono richiesti binding OpenSim compatibili con la versione del modello e con l’interprete Python attivo.
