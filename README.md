# opensim-models

Package Python per costruire e comporre modelli OpenSim. `OpenSimModel` è una facade generica su un modello OpenSim (caricamento, coordinate, marker, muscoli, scaling, visualizzazione, composizione di più modelli, importazione da CAD). Le sottoclassi concrete oggi disponibili sono `User`, un utente antropometrico costruito a partire dal modello full-body di Rajagopal-Lai-Uhlrich e dai riferimenti ANSUR II, già scalato e pronto per analisi biomeccaniche, simulazioni e manipolazione della postura, e `Screen`, un pannello plexiglass parametrico (es. per rappresentare un monitor in scena).

## Contenuto del progetto

```text
src/opensim_models/
	model.py                       # OpenSimModel: facade generica, show(), composizione di modelli, OpenSimModel.from_step
	operators.py                   # add_*/remove_* generici per corpi, giunti, forze/muscoli, marker, vincoli, ...
	_cad_import.py                 # interno: lettura STEP/STP e generazione mesh per OpenSimModel.from_step
	models/
		user/
			user.py                    # User(OpenSimModel): scaling antropometrico e setter di postura
			_data.py                   # interno: caricamento ANSUR e percentili
			_mapping.py                # interno: mappa ANSUR -> corpi OpenSim
			assets/
				ansur_ref.csv           # riferimenti antropometrici ANSUR II
				rajagopalaiulrich2023.osim  # modello OpenSim base di User
				meshes/*.vtp            # mesh per il rendering di User
		screen/
			screen.py                  # Screen(OpenSimModel): pannello plexiglass parametrico
			assets/meshes/              # mesh generata automaticamente per Screen (vedi sotto)
tests/
	test_model.py                  # test esaustivi di OpenSimModel (facade + composizione)
	test_operators.py              # test esaustivi di opensim_models.operators
	test_user.py                   # test esaustivi di User (dati ANSUR, scaling, postura)
	test_screen.py                 # test esaustivi di Screen (dimensionamento, posa, mesh)
	test_cad_import.py             # test di OpenSimModel.from_step (richiede pythonocc-core)
```

Ogni modello specifico (oggi `User` e `Screen`) vive nella propria sottocartella sotto `models/`, con il proprio codice e i propri asset. Nuovi modelli (es. un attrezzo da palestra) si aggiungono allo stesso modo, come ulteriori sottoclassi di `OpenSimModel`.

**Superficie pubblica.** Per ogni modello, l'unico simbolo importabile è la sua sottoclasse di `OpenSimModel` (`User`, `Screen`): `from opensim_models import OpenSimModel, User, Screen` è l'API pubblica principale del package. Tutto il resto -- dataset ANSUR, funzioni di risoluzione dei percentili, mappe di scaling, lettura CAD -- è dettaglio implementativo del modello che lo usa: vive in moduli non riesportati dai vari `__init__.py` (per `User`, i moduli con prefisso `_`, come `_data.py` e `_mapping.py`) e non è pensato per essere importato direttamente. Fa eccezione `opensim_models.operators` (vedi sotto): un modulo di utilità pensato per essere importato direttamente (`from opensim_models import operators`), non riesportato al livello superiore del package per restare distinto dalle sottoclassi di modello.

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

Per costruire un modello a partire da un file CAD (`OpenSimModel.from_step`, vedi sotto) serve in aggiunta `pythonocc-core` (binding Python di OpenCascade), anch'esso non installabile in modo affidabile via pip:

```powershell
conda install -c conda-forge pythonocc-core
```

Anche questa dipendenza è caricata solo al momento della chiamata a `from_step`: il resto del package funziona normalmente senza `pythonocc-core` installato.

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

## Architettura: Model, State e propagazione

OpenSim separa la *struttura* di un modello (`opensim.Model`: corpi, giunti, muscoli e le loro proprietà) dalla sua *condizione istantanea* (`opensim.State`: valori delle coordinate, velocità, attivazioni muscolari, e una cache di tutto ciò che ne viene derivato -- posizioni dei corpi, forze...). `OpenSimModel` espone entrambi direttamente come attributi: `model.model` e `model.state`.

Da questa separazione derivano tre categorie di operazioni, ciascuna con un comportamento diverso:

1. **Setter "grezzi" sullo stato** -- `set_coordinate_degrees`, `set_coordinate_speed_degrees`: scrivono immediatamente il valore in `state`, ma **non propagano nulla**. Rileggerli subito dopo va bene (`coordinate_degrees`/`coordinate_speed_degrees` leggono lo stesso valore grezzo), ma qualunque quantità *derivata* (posizione di un corpo, lunghezza di un muscolo, forze...) resta non aggiornata finché non chiami `update_state()`. Molte query di OpenSim su quantità derivate lo verificano da sole e sollevano un errore esplicito (`RuntimeError`) se lo stato non è stato propagato, invece di restituire silenziosamente un valore obsoleto -- ma non è garantito per ogni query, quindi non affidarti a quell'errore come unico controllo.
2. **`update_state()`** -- propaga i valori impostati fino allo stage OpenSim `Dynamics` (posizione, velocità, equilibrio muscolare, forze). Va chiamato una volta dopo un blocco di più modifiche (più efficiente che propagare ad ogni singola chiamata), o comunque prima di leggere una quantità derivata. Non risolve un'eventuale coordinata accoppiata (es. rotula-ginocchio) né le accelerazioni -- vedi i limiti più sotto.
3. **Setter di proprietà strutturali** -- `set_body_mass`, o proprietà OpenSim impostate direttamente sui componenti (es. `muscle.set_ignore_tendon_compliance(True)`): richiedono che OpenSim ricostruisca l'intero sistema (`initSystem()`), operazione che altrimenti azzererebbe ogni coordinata al valore di default del file `.osim`. `reinitialize()` fa questa ricostruzione preservando postura, velocità e la coerenza delle coordinate accoppiate; `set_body_mass` la richiama automaticamente.

In sintesi: i setter registrano l'intento, `update_state()` va chiamato prima di leggere qualunque cosa di derivato, `reinitialize()` (o un setter come `set_body_mass` che lo richiama da solo) va chiamato dopo una modifica strutturale.

### Limiti noti di `update_state()`

- **Non risolve un `CoordinateCouplerConstraint`** (es. rotula agganciata alla flessione del ginocchio): realizzare gli stage OpenSim non fa ricalcolare a Simbody il valore di una coordinata dipendente da quella indipendente -- servirebbe `Model.assemble()`, che però non è affidabile su ogni modello muscolo-scheletrico (stesso tipo di crash nativo descritto sotto per `Acceleration`). Se una coordinata accoppiata deve riflettere la nuova postura per un uso reale (non solo per l'export), usa `reinitialize()`, che la ricalcola passando dai valori di default (non richiede l'assemblaggio).
- **Si ferma allo stage `Dynamics`, non realizza `Acceleration`**: su alcuni modelli muscolo-scheletrici (incluso quello di `User`) `Model.realizeAcceleration` causa un crash nativo irreversibile del processo Python (non un'eccezione catturabile). Se ti servono le accelerazioni, valuta tu stesso il rischio chiamando `model.model.realizeAcceleration(model.state)` direttamente.

## Modificare la postura

Gli angoli dell'API pubblica sono sempre espressi in gradi. La conversione in radianti viene effettuata internamente prima di chiamare OpenSim. I setter di postura scrivono solo il valore grezzo (vedi sopra): chiama `update_state()` prima di leggere qualunque quantità derivata dalla nuova postura.

```python
user.set_right_hip_flexionextension(25.0)
user.set_left_hip_adduction(10.0)
user.set_right_knee_flexionextension(40.0)
user.set_left_ankle_flexiondorsiflexion(5.0)
user.set_right_shoulder_flexion(30.0)
user.set_left_elbow_flexion(90.0)
user.set_right_wrist_deviation(12.0)
user.set_lumbar_extension(8.0)
user.update_state()

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

## Velocità, massa e ricostruzione dello stato

Analogamente alla posizione, anche la velocità di una coordinata ha un setter/getter dedicato (ereditato da `OpenSimModel`); scrive solo il valore grezzo, come `set_coordinate_degrees` (vedi l'architettura sopra):

```python
user.set_coordinate_speed_degrees("hip_flexion_r", 45.0)
user.update_state()
print(user.coordinate_speed_degrees("hip_flexion_r"))  # 45.0
```

Anche la massa di un corpo ha un setter/getter dedicato -- ma qui il meccanismo è diverso da `update_state()`: la massa entra nella matrice del sistema multibody, costruita da OpenSim una sola volta in `initSystem()`, quindi il setter richiama automaticamente `reinitialize()` (vedi sotto) invece di limitarsi a scrivere il valore:

```python
user.set_body_mass("tibia_r", 5.0)
print(user.body_mass("tibia_r"))  # 5.0
```

Più in generale, qualunque proprietà OpenSim che richieda una ricostruzione strutturale del sistema (es. `ignore_tendon_compliance` su un muscolo) va seguita da `reinitialize()`, non da `user.model.initSystem()` diretto (che azzererebbe la postura):

```python
for muscle_name in ["soleus_r", "gasmed_r"]:
    user.muscle(muscle_name).set_ignore_tendon_compliance(True)

user.reinitialize()  # invece di user.model.initSystem()
```

`export()`, `scale_bodies()`, `add_model()` e `remove_model()` (vedi sotto) applicano internamente la stessa logica di preservazione di `reinitialize()`, quindi la postura sopravvive anche a scaling, export e composizione di modelli senza bisogno di intervento manuale.

## Copiare un modello

`copy()` (ereditato da `OpenSimModel`) restituisce una copia indipendente di qualunque modello, dello stesso tipo dell'originale, senza che le sottoclassi (`User`, `Screen`, o future sottoclassi) debbano implementare un proprio override: copia tutti gli attributi dell'istanza e sostituisce solo le parti che devono restare indipendenti (il modello OpenSim sottostante, il suo stato, le collezioni di bookkeeping mutabili), preservando la postura corrente:

```python
user_copy = user.copy()
assert type(user_copy) is type(user)
assert user_copy.model is not user.model

user_copy.set_right_hip_flexionextension(45.0)  # non tocca l'originale
```

## Aggiungere/rimuovere componenti (`operators`)

Il modulo `opensim_models.operators` fornisce funzioni per aggiungere/rimuovere corpi, giunti, forze (inclusi i muscoli), marker, vincoli, controller, geometrie di contatto e probe da un `OpenSimModel` -- tutto ciò che è gestito da un `opensim.Model`:

```python
from opensim_models import OpenSimModel, operators

model = OpenSimModel(model_path=None)

with model.structural_change():
    body = operators.add_body(model, "b1", mass=2.0, inertia=(1, 1, 1, 0, 0, 0))
    operators.add_joint(
        model, model.opensim.FreeJoint("b1_to_ground", model.model.getGround(), body)
    )

print(model.bodies.getSize())  # 1
```

Aggiungere o rimuovere un componente è un cambiamento *strutturale*: rende `model.state` immediatamente non valido (non solo i suoi valori di default, ma proprio l'oggetto stato -- leggerlo crasha il processo invece di sollevare un'eccezione catturabile), perché la struttura del sistema è cambiata sotto di esso. Per questo ogni funzione `add_*`/`remove_*` di default (`reinitialize=False`) esegue solo la modifica strutturale, senza toccare lo stato:

- per una singola modifica autosufficiente, passa `reinitialize=True` (es. `operators.add_marker(model, marker, reinitialize=True)`);
- per un gruppo di modifiche correlate (un corpo e il giunto che lo collega; un giunto e il corpo che rimuove insieme) usa `model.structural_change()`, che sincronizza la postura corrente **prima** che inizi il blocco (quando lo stato è ancora valido) e ricostruisce il sistema **una sola volta** all'uscita, preservando quella postura. Non leggere `model.state` (direttamente, o tramite un accessor di coordinate/marker/muscoli) dentro il blocco.

`add_body` costruisce direttamente un `opensim.Body` (unico tipo con un costruttore universale); per forze/muscoli, marker, vincoli, controller, geometrie di contatto e probe -- che in OpenSim hanno costruttori molto diversi tra loro -- le funzioni `add_*` si limitano ad agganciare al modello un componente già costruito dal chiamante con l'API nativa di OpenSim (es. `opensim.Millard2012EquilibriumMuscle(...)`). È disponibile anche una coppia generica `add_component(model, kind, component)`/`remove_component(model, kind, name)` per qualunque categoria (`"body"`, `"joint"`, `"force"`, `"marker"`, `"constraint"`, `"controller"`, `"contact_geometry"`, `"probe"`), utile per codice generico che non conosce la categoria in anticipo.

`add_body` accetta anche `mesh_files` per collegare una o più mesh già esistenti (`.vtp`/`.stl`/`.obj`) come geometria del corpo, registrandone automaticamente le cartelle per la ricerca geometrica di OpenSim:

```python
body = operators.add_body(model, "part1", mass=1.5, mesh_files="assets/part1.stl")
```

### Giunti nominati

Oltre al generico `add_joint(model, joint)` (che accetta un giunto già costruito con l'API nativa, per qualunque tipo, incluso `CustomJoint`), sono disponibili funzioni dedicate per i tipi più comuni -- `add_free_joint` (6 gdl), `add_pin_joint` (1 gdl rotazionale, attorno al proprio asse Z), `add_ball_joint` (3 gdl rotazionali), `add_slider_joint` (1 gdl traslazionale, lungo il proprio asse X) e `add_weld_joint` (0 gdl, vincolo rigido) -- che costruiscono il giunto per te a partire da posizione e orientamento:

```python
operators.add_pin_joint(
    model, "knee", child_body,
    position=(0.0, -0.4, 0.0),      # nel parent_frame (default: ground), in metri
    orientation_deg=(0.0, 0.0, 90.0),  # angoli di Eulero X-Y-Z, in gradi
)
```

`position` è la collocazione del giunto nel `parent_frame` (default `ground`); con il corpo figlio costruito secondo la convenzione `mass_center=(0, 0, 0)` (vedi `add_body`), coincide con la posizione del baricentro del corpo nello spazio. `orientation_deg` sono gli angoli di Eulero X-Y-Z (body-fixed), in gradi, attorno agli assi del `parent_frame`. `child_position`/`child_orientation_deg` offrono lo stesso controllo sul lato del corpo figlio (di norma lasciati a zero).

### Corpi con forma primitiva (box, cilindro, sfera)

`add_box_body`, `add_cylinder_body` e `add_sphere_body` costruiscono in un solo passaggio un corpo di forma nota, il giunto (di default un `WeldJoint`, cioè fisso) che lo posiziona/orienta, e la relativa geometria:

```python
body, joint = operators.add_cylinder_body(
    model, "post", radius=0.02, height=0.5,
    density=2700.0,                    # es. alluminio, kg/m^3
    position=(1.0, 0.0, 0.0),          # baricentro nel ground, in metri
    orientation_deg=(0.0, 0.0, 90.0),  # angoli di Eulero X-Y-Z, in gradi
    joint_type="free",                 # invece del weld di default
    reinitialize=True,
)
```

Massa e tensore d'inerzia centrale sono calcolati analiticamente da dimensioni e `density` (default `1000.0` kg/m³, un segnaposto generico come per `OpenSimModel.from_step`), quindi corrispondono sempre esattamente a quanto disegnato. Per default (`mesh=False`) viene collegata una geometria nativa leggera (`opensim.Brick`/`Cylinder`/`Sphere`, senza scrivere alcun file); passando `mesh=True` (e indicando `mesh_dir`) viene invece generata e collegata una vera mesh STL, utile per un export portabile del modello. Il cilindro ha l'asse lungo il proprio Y (come la convenzione nativa di `opensim.Cylinder`): usa `orientation_deg` per orientarlo diversamente. Come per le altre funzioni del modulo, `reinitialize=False` (default) permette di aggiungere più forme primitive in un unico batch dentro `model.structural_change()`.

## Dati ANSUR risolti

Il caricamento del CSV ANSUR e il calcolo dei percentili (`load_ansur`, `resolve_reference`) sono dettagli implementativi interni di `User`, non parte della superficie pubblica del package (vedi sopra). I valori risolti restano comunque accessibili dopo aver costruito un `User`, tramite la proprietà `anthropometry`:

```python
user = User("F", percentile=75.0)

reference = user.anthropometry
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

Il file esportato con `user.export(...)` contiene il modello scalato con la postura corrente. `export()` copia inoltre automaticamente ogni mesh referenziata dai corpi del modello in una cartella `Geometry/` accanto al file `.osim` esportato (la convenzione di nome che OpenSim/Simbody cercano automaticamente accanto a un modello), così l'esportazione è portabile anche senza le cartelle di geometria originali (`models/user/assets/meshes/` per `User`, la cartella di `from_step` per un modello CAD).

## Creare uno schermo (Screen)

`Screen` è un pannello rigido in plexiglass, spesso 1 mm, pensato per rappresentare un monitor/schermo nella scena:

```python
from opensim_models import Screen

screen = Screen()  # 22", 16:9, centrato nell'origine, verticale (angle_deg=90)

print(screen.width_mm, screen.height_mm)  # None, None: dimensione derivata dalla diagonale
screen.show()
```

Le dimensioni si ottengono in due modi alternativi, con priorità automatica: se `width_mm` e `height_mm` sono *entrambi* impostati vincono loro; altrimenti (compreso il default, con entrambi `None`) la dimensione viene calcolata dalla diagonale in pollici (`inches`, default `22`) e dal rapporto di forma (`ratio`, default `"16:9"`):

```python
screen_esplicito = Screen(width_mm=600.0, height_mm=340.0)
screen_diagonale = Screen(inches=27.0, ratio="21:9")
```

`center_x`/`center_y`/`center_z` posizionano il centro del pannello nel sistema di riferimento del ground (metri); `angle_deg` ne definisce l'inclinazione rispetto al ground: `0` disteso a terra, `90` (default) verticale, come un monitor appoggiato su un piano orizzontale.

Ogni parametro del costruttore ha una property in lettura (`width_mm`, `height_mm`, `inches`, `ratio`, `center_x`, `center_y`, `center_z`, `angle_deg`) e un setter dedicato (`set_width_mm`, `set_height_mm`, `set_inches`, `set_ratio`, `set_center_x`, `set_center_y`, `set_center_z`, `set_angle_deg`). Ogni setter ricostruisce il corpo OpenSim, la mesh e il giunto verso ground con i parametri aggiornati:

```python
screen.set_angle_deg(0)       # ora disteso sul piano orizzontale
screen.set_width_mm(600.0)
screen.set_height_mm(340.0)   # passa in modalità dimensioni esplicite solo una volta impostate entrambe
```

Massa e tensore d'inerzia del pannello derivano dal suo volume (larghezza × altezza × 1 mm) assumendo una densità da plexiglass/PMMA (`1180 kg/m³`); una mesh a forma di parallelepipedo viene generata automaticamente e salvata in `models/screen/assets/meshes/screen_panel.stl`, rigenerata a ogni cambio di dimensione. Il pannello è un unico `opensim.Body` ("screen_panel") saldato al ground con un `WeldJoint` (nessun grado di libertà): la sua posa è interamente determinata da `center_x`/`center_y`/`center_z`/`angle_deg`.

## Comporre più modelli

Due o più `OpenSimModel` (ad esempio un `User` e uno `Screen`) possono essere combinati in un unico modello OpenSim esportabile:

```python
combined = user_model + screen_model
combined.export("combined.osim")
```

`combined` è un `OpenSimModel` generico che contiene tutti i componenti di entrambi gli operandi (corpi, giunti, muscoli/forze, marker, vincoli); nessuno dei due operandi originali viene modificato. Se un componente del secondo modello ha lo stesso nome di uno già presente nel primo, viene rinominato automaticamente con un prefisso (il nome della classe del modello, es. `screen_screen_panel`, oppure un prefisso esplicito tramite `name=`). I giunti agganciati al `ground` nei modelli sorgente restano agganciati al ground condiviso del modello combinato, così i due modelli mantengono la propria collocazione di default.

Per comporre più di due modelli, `+` si può concatenare (`a + b + c`), oppure si può operare in place su un modello esistente:

```python
scene = OpenSimModel(model_path=None)
scene.add_model(user_model)
scene.add_model(screen_model, name="screen")
...
scene.remove_model(screen_model)  # torna allo stato precedente
```

`add_model`, `remove_model`, `+` e la sua forma riflessa richiedono sempre che l'altro operando sia un `OpenSimModel`: in caso contrario sollevano `TypeError`. `remove_model` richiede che il modello indicato sia stato effettivamente aggiunto con `add_model` in precedenza, altrimenti solleva `ValueError`.

## Costruire un modello da CAD (.step/.stp)

`OpenSimModel.from_step` costruisce un modello direttamente da un assieme CAD in formato STEP. Per default (`as_one_object=True`) tutti i solidi del file vengono saldati in un unico corpo OpenSim, con massa/inerzia combinate; con `as_one_object=False` genera invece un corpo per ogni solido trovato nel file, come nelle versioni precedenti:

```python
from opensim_models import OpenSimModel

model = OpenSimModel.from_step("assieme.step", density=2700.0)  # es. alluminio, kg/m^3

print(model.bodies.getSize())   # 1: tutti i solidi combinati in un unico corpo
model.show()
model.export("assieme.osim")

# Un corpo per ogni solido/parte, come nel comportamento storico:
model_per_parte = OpenSimModel.from_step("assieme.step", density=2700.0, as_one_object=False)
print(model_per_parte.bodies.getSize())   # un body per solido/parte nominata nel file
```

Per ogni solido:

- il nome del corpo OpenSim (in modalità `as_one_object=False`) viene ricavato dal nome della parte/prodotto nel file STEP (quando presente), altrimenti da un nome generico (`body_0`, `body_1`, ...); in modalità `as_one_object=True` il corpo unico prende il nome del file STEP;
- massa e tensore d'inerzia sono calcolati dal volume del solido moltiplicato per la densità (`density`, oppure per parte tramite `densities={"nome_parte": ...}`); un file STEP puro raramente porta informazioni di materiale, quindi il valore di default (`1000.0` kg/m³) è solo un segnaposto generico. In modalità `as_one_object=True` la massa, il baricentro e il tensore d'inerzia di ciascun solido vengono combinati (teorema degli assi paralleli) in un'unica massa/inerzia per il corpo combinato;
- viene generata e scritta su disco una mesh triangolare del solido (in `mesh_dir`, di default una cartella `{nome_file}_meshes` accanto al file STEP) e collegata al corpo come geometria; in modalità `as_one_object=False` è centrata sul baricentro del singolo solido, in modalità `as_one_object=True` sul baricentro combinato dell'intero assieme (più mesh, una per solido, collegate allo stesso corpo);
- l'unità dichiarata nel file STEP (millimetri, centimetri, pollici, ...) viene convertita automaticamente in metri.

Un assieme CAD non contiene alcuna informazione cinematica: per poter caricare e simulare subito il modello, ogni corpo viene per default collegato al `ground` con un `FreeJoint` (6 gradi di libertà) posizionato nella collocazione originale del CAD (`add_free_joints=True`). Questi giunti sono un default di comodo, non la catena cinematica reale dell'assieme: vanno sostituiti con i giunti corretti prima di usare il modello per una simulazione dinamica. Con `add_free_joints=False` i corpi vengono aggiunti senza giunti; sarà poi necessario collegarli manualmente e richiamare `model.model.initSystem()`.

## Test

Per eseguire l'intera suite (test statistici sui dati ANSUR e test di integrazione OpenSim):

```powershell
python -m pytest -q
```

- `tests/test_model.py` copre `OpenSimModel` in modo esaustivo: caricamento (da file, vuoto, file mancante), sblocco delle coordinate, accessori nominati, gestione di coordinate (posizione, velocità)/marker/muscoli/massa dei corpi, `update_state()` (propagazione a quantità derivate, comportamento "grezzo" dei setter), `reinitialize()` (inclusa la coerenza di coordinate accoppiate da un `CoordinateCouplerConstraint`), `copy()` (indipendenza del modello copiato, preservazione di postura e di attributi delle sottoclassi), scaling, export (inclusa la copia delle mesh in `Geometry/`), cartelle di geometria e l'intera composizione di modelli (`add_model`, `remove_model`, `__add__`, `__radd__`, rinomina automatica sulle collisioni, preservazione della postura degli operandi, controlli di tipo).
- `tests/test_user.py` copre `User` in modo esaustivo: caricamento e validazione dei dati ANSUR (eseguibili anche senza OpenSim installato), risoluzione di percentile/altezza, scaling antropometrico, ogni singolo setter di postura, e l'integrazione con la facade ereditata da `OpenSimModel`.
- `tests/test_screen.py` copre `Screen` in modo esaustivo: dimensionamento (esplicito, da diagonale, priorità e fallback tra i due), posa (`center_*`/`angle_deg`), struttura del modello (un corpo, un `WeldJoint`), rigenerazione della mesh sui setter e registrazione della cartella di geometria.
- `tests/test_cad_import.py` copre `OpenSimModel.from_step`: massa/inerzia calcolate correttamente da un solido di riferimento (con conversione di unità), generazione della mesh, giunti verso ground di default, combinazione di più solidi in un unico corpo (`as_one_object`, di default e disattivata) e relativi errori (file mancante, STEP senza solidi).
- `tests/test_operators.py` copre `opensim_models.operators`: `add_component`/`remove_component` generici e i relativi errori, i wrapper nominati per corpi/giunti/forze-muscoli/marker/vincoli, i costruttori di giunto nominati (gradi di libertà, posizione/orientamento), i corpi a forma primitiva (massa/inerzia analitiche, geometria nativa vs mesh generata, tipo di giunto, batching), il collegamento di mesh esistenti a un corpo, l'uso di `structural_change()` per un batch di modifiche correlate (corpo+giunto) e la preservazione della postura delle coordinate non toccate dalla modifica strutturale.

I test che richiedono i binding OpenSim vengono saltati automaticamente se il modulo `opensim` non è importabile; quelli di `test_cad_import.py` vengono saltati se `pythonocc-core` non è importabile; i test sui soli dati ANSUR restano eseguibili in ogni caso.

## Limiti e note

- Lo scaling è completo a livello di pipeline OpenSim, ma la qualità antropometrica dipende dalla corrispondenza tra misura ANSUR e segmento.
- Le misure senza corrispondenza diretta usano il rapporto di statura come fallback esplicito.
- L'altezza richiesta viene trasformata nel percentile ANSUR equivalente; per questo `user.height` può differire leggermente dall'input originale.
- La composizione di modelli (`add_model`/`__add__`) è pensata per scheletri/oggetti indipendenti agganciati al ground: non offre (ancora) un modo per saldare un modello a un body specifico dell'altro.
- Sono richiesti binding OpenSim compatibili con la versione del modello e con l'interprete Python attivo.
- `OpenSimModel.from_step` non deduce alcuna gerarchia cinematica dal file CAD (un file STEP non la contiene): i `FreeJoint` generati di default vanno sostituiti con i giunti reali dell'assieme prima di affidarsi alla dinamica del modello.
- La densità usata da `from_step` è un valore generico in assenza di dati materiale nel file STEP: per una massa/inerzia fisicamente corrette va passata esplicitamente (globalmente o per parte).
- Ogni setter di `Screen` ricostruisce da zero corpo, mesh e giunto: economico per un singolo pannello, ma non pensato per essere chiamato ad alta frequenza (es. in un loop di animazione).
