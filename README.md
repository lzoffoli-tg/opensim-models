# opensim-models

Package Python per costruire e comporre modelli OpenSim. `OpenSimModel` è una facade generica su un modello OpenSim (caricamento, coordinate, marker, muscoli, scaling, visualizzazione, composizione di più modelli). `User` è un modello specifico che la specializza: un utente antropometrico costruito a partire dal modello full-body di Rajagopal-Lai-Uhlrich e dai riferimenti ANSUR II, già scalato e pronto per analisi biomeccaniche, simulazioni e manipolazione della postura.

## Contenuto del progetto

```text
src/opensim_models/
	model.py                       # OpenSimModel: facade generica, show(), composizione di modelli
	models/
		user/
			user.py                    # User(OpenSimModel): scaling antropometrico e setter di postura
			data.py                    # caricamento ANSUR e percentili
			mapping.py                 # mappa ANSUR -> corpi OpenSim
			assets/
				ansur_ref.csv           # riferimenti antropometrici ANSUR II
				rajagopalaiulrich2023.osim  # modello OpenSim base di User
				meshes/*.vtp            # mesh per il rendering di User
tests/
	test_model.py                  # test esaustivi di OpenSimModel (facade + composizione)
	test_user.py                   # test esaustivi di User (dati ANSUR, scaling, postura)
```

Ogni modello specifico (oggi solo `User`) vive nella propria sottocartella sotto `models/`, con il proprio codice e i propri asset. Nuovi modelli (es. un attrezzo da palestra) si aggiungono allo stesso modo, come ulteriori sottoclassi di `OpenSimModel`.

Il modello di `User` referenzia 81 mesh VTP, tutte incluse nella cartella `models/user/assets/meshes/`. Sono presenti anche quattro alias aggiuntivi per femori e tibie.

## Installazione

Il package richiede Python 3.10 o superiore, NumPy e Pandas. I binding Python di OpenSim sono una dipendenza nativa opzionale, necessaria solo per costruire/scalare/visualizzare un modello effettivo:

```powershell
python -m pip install -e ".[test,opensim]"
```

Su Windows è comunque consigliato un ambiente Conda compatibile con la versione OpenSim utilizzata, se l'extra `opensim` non è installabile via pip:

```powershell
conda install -c opensim-org opensim
```

L'importazione di `opensim_models` non carica OpenSim immediatamente: il binding è richiesto solo quando si costruisce un `OpenSimModel` (o una sua sottoclasse come `User`); in assenza di un ambiente configurato viene sollevato un `RuntimeError` esplicito. Alla prima costruzione, il `PATH` nativo necessario al visualizzatore Simbody su Windows (le DLL nella cartella Conda `Library/bin`) viene sistemato automaticamente: non è richiesta nessuna configurazione manuale aggiuntiva.

## Creare un utente

### Percentile predefinito

Passando solo il sesso, viene usato il 50° percentile per tutte le misure, statura inclusa:

```python
from opensim_models import User

male = User("M")
female = User("F")

print(male.gender)       # "M"
print(male.percentile)   # 50.0
print(male.height)       # statura risolta in centimetri
```

`gender` accetta solo `"M"` e `"F"`.

### Percentile esplicito

```python
large_male = User("M", percentile=75)

print(large_male.percentile)  # 75.0
print(large_male.height)      # statura al 75° percentile maschile
```

Il percentile deve appartenere all'intervallo inclusivo `[0.1, 99.9]`. Il calcolo usa direttamente `numpy.percentile` per ogni misura numerica ANSUR: non viene scelto un singolo soggetto e non viene effettuata interpolazione tra misure antropometriche.

### Altezza esplicita

L'altezza è espressa in centimetri. Il package calcola il percentile empirico della statura richiesta nel gruppo ANSUR del sesso indicato e applica quel percentile a tutte le misure:

```python
user = User("M", height=175.0)

print(user.height)       # statura ANSUR risolta in centimetri
print(user.percentile)   # percentile empirico corrispondente a 175 cm
```

Quando `height` è presente, determina il percentile effettivo e ha precedenza sul valore passato in `percentile`. L'altezza deve essere positiva e compresa nel range osservato dal dataset ANSUR.

## Scaling antropometrico

Durante la costruzione di `User` vengono eseguiti questi passaggi:

1. caricamento e normalizzazione del CSV ANSUR;
2. filtraggio per sesso;
3. calcolo del percentile comune per tutte le misure;
4. confronto con il riferimento del 50° percentile dello stesso sesso;
5. costruzione dei fattori `(x, y, z)` per i corpi OpenSim;
6. applicazione tramite `OpenSim.Model.scale` e `ScaleSet` (ereditata da `OpenSimModel.scale_bodies`).

La scalatura nativa OpenSim aggiorna in modo coordinato corpi, geometrie, frame articolari, marker e percorsi muscolari. La mappa usa direttamente le misure ANSUR disponibili per bacino, tronco, femori, tibie, piedi, braccia, avambracci e mani. Per assi o segmenti senza una misura ANSUR sufficientemente diretta viene usato il rapporto di statura come fallback deterministico.

Il file `.osim` originale e il CSV non vengono modificati. Ogni istanza possiede un modello OpenSim indipendente:

```python
user_50 = User("M", percentile=50)
user_75 = User("M", percentile=75)

assert user_50.model is not user_75.model
```

## Accesso al modello OpenSim

`User` eredita da `OpenSimModel` l'intera facade sul modello OpenSim sottostante. Le collezioni principali sono disponibili come proprietà:

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
gluteus = user.muscle("glmax1_r")
marker = user.marker("RASI")
coordinate = user.coordinate("hip_flexion_r")
```

Il modello base di `User` contiene 22 corpi, 22 giunti, 80 muscoli, 66 marker e 39 coordinate.

## Modificare la postura

Gli angoli dell'API pubblica sono sempre espressi in gradi. La conversione in radianti viene effettuata internamente prima di chiamare OpenSim.

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

I setter che agiscono su una coordinata con lato (anca, ginocchio, caviglia, piede, spalla, gomito, polso, avambraccio) sono disponibili in coppia `set_left_*`/`set_right_*`. Sono disponibili per:

- flessione/estensione ed adduzione/abduzione e rotazione dell'anca;
- flessione/estensione del ginocchio;
- flessione/dorsiflessione della caviglia;
- inversione subtalare e flessione MTP;
- flessione, adduzione e rotazione della spalla;
- flessione del gomito;
- flessione e deviazione del polso;
- pronazione/supinazione dell'avambraccio.

I setter lombari non hanno lato e restano invariati: estensione, inclinazione laterale e rotazione lombare.

È inoltre disponibile il setter generico (ereditato da `OpenSimModel`), utilizzabile su qualunque coordinata del modello:

```python
user.set_coordinate_degrees("arm_rot_r", 15.0)
angle = user.coordinate_degrees("arm_rot_r")
```

Le coordinate bloccate dal modello, come le coordinate subtalare e MTP di questa versione, vengono sbloccate automaticamente al caricamento; un blocco impostato esplicitamente con `set_coordinate_locked` fa invece sollevare `ValueError` al relativo setter, invece di ignorare silenziosamente il valore.

## Riferimenti ANSUR senza OpenSim

Per usare soltanto il caricamento dati e i percentili, senza bisogno dei binding OpenSim:

```python
from opensim_models.models.user import load_ansur, resolve_reference

data = load_ansur()
print(data.shape)

reference = resolve_reference(gender="F", percentile=75.0)
print(reference.height_cm)
print(reference.values["footlength"])
```

Le misure lineari ANSUR restano nelle unità sorgente, prevalentemente millimetri. `stature_m` è in metri; `height_cm` è una proprietà di comodo in centimetri.

## Rendering

Ogni modello registra automaticamente la propria cartella di mesh: per `User` non serve indicare alcun percorso di geometria.

```python
user = User("M")
user.show()
```

`show()` (ereditato da `OpenSimModel`) apre il visualizzatore nativo Simbody con la postura corrente. Per aggiungere ulteriori cartelle di geometria (ad esempio per un modello composto, vedi sotto) si può passare `geometry_path` esplicitamente, oppure registrarle in anticipo con `add_geometry_directory(...)`.

Il file esportato con `user.export(...)` contiene il modello scalato con la postura corrente; le mesh originali restano nella cartella `models/user/assets/meshes/`.

## Comporre più modelli

Due o più `OpenSimModel` (ad esempio un `User` e, in futuro, un modello di un attrezzo da palestra) possono essere combinati in un unico modello OpenSim esportabile:

```python
combined = user_model + equipment_model
combined.export("combined.osim")
```

`combined` è un `OpenSimModel` generico che contiene tutti i componenti di entrambi gli operandi (corpi, giunti, muscoli/forze, marker, vincoli); nessuno dei due operandi originali viene modificato. Se un componente del secondo modello ha lo stesso nome di uno già presente nel primo, viene rinominato automaticamente con un prefisso (il nome della classe del modello, es. `equipment_pelvis`, oppure un prefisso esplicito tramite `name=`). I giunti agganciati al `ground` nei modelli sorgente restano agganciati al ground condiviso del modello combinato, così i due modelli mantengono la propria collocazione di default.

Per comporre più di due modelli, `+` si può concatenare (`a + b + c`), oppure si può operare in place su un modello esistente:

```python
scene = OpenSimModel(model_path=None)
scene.add_model(user_model)
scene.add_model(equipment_model, name="equipment")
...
scene.remove_model(equipment_model)  # torna allo stato precedente
```

`add_model`, `remove_model`, `+` e la sua forma riflessa richiedono sempre che l'altro operando sia un `OpenSimModel`: in caso contrario sollevano `TypeError`. `remove_model` richiede che il modello indicato sia stato effettivamente aggiunto con `add_model` in precedenza, altrimenti solleva `ValueError`.

## Test

Per eseguire l'intera suite (test statistici sui dati ANSUR e test di integrazione OpenSim):

```powershell
python -m pytest -q
```

- `tests/test_model.py` copre `OpenSimModel` in modo esaustivo: caricamento (da file, vuoto, file mancante), sblocco delle coordinate, accessori nominati, gestione di coordinate/marker/muscoli, scaling, export, cartelle di geometria e l'intera composizione di modelli (`add_model`, `remove_model`, `__add__`, `__radd__`, rinomina automatica sulle collisioni, controlli di tipo).
- `tests/test_user.py` copre `User` in modo esaustivo: caricamento e validazione dei dati ANSUR (eseguibili anche senza OpenSim installato), risoluzione di percentile/altezza, scaling antropometrico, ogni singolo setter di postura, e l'integrazione con la facade ereditata da `OpenSimModel`.

I test che richiedono i binding OpenSim vengono saltati automaticamente se il modulo `opensim` non è importabile; i test sui soli dati ANSUR restano eseguibili in ogni caso.

## Limiti e note

- Lo scaling è completo a livello di pipeline OpenSim, ma la qualità antropometrica dipende dalla corrispondenza tra misura ANSUR e segmento.
- Le misure senza corrispondenza diretta usano il rapporto di statura come fallback esplicito.
- L'altezza richiesta viene trasformata nel percentile ANSUR equivalente; per questo `user.height` può differire leggermente dall'input originale.
- La composizione di modelli (`add_model`/`__add__`) è pensata per scheletri/oggetti indipendenti agganciati al ground: non offre (ancora) un modo per saldare un modello a un body specifico dell'altro.
- Sono richiesti binding OpenSim compatibili con la versione del modello e con l'interprete Python attivo.
