# opensim-models

Package Python per costruire e comporre modelli OpenSim. `OpenSimModel` è una facade generica su un modello OpenSim (caricamento, coordinate, marker, muscoli, scaling, rotazione/traslazione rigida, visualizzazione (con riproduzione di una simulazione), composizione di più modelli, importazione da CAD) -- un **container**, nel senso CAD del termine. `User`, oggi la sua unica sottoclasse concreta, è un utente antropometrico costruito a partire dal modello full-body di Rajagopal-Lai-Uhlrich e dai riferimenti ANSUR II, già scalato e pronto per analisi biomeccaniche, simulazioni e manipolazione della postura.

`Screen` e `Box` sono invece **componenti** (parti, sempre nel senso CAD): un pannello plexiglass parametrico (es. per rappresentare un monitor in scena) e un parallelepipedo rigido generico (es. per un ingombro o un componente di un attrezzo). A differenza di `User`, non sono `OpenSimModel`: non hanno un proprio `show()`/`export()`, e per essere visualizzati vanno prima aggiunti a un container (`model + screen`, `model + box`, vedi "Comporre più modelli").

## Contenuto del progetto

```text
src/opensim_models/
	model.py                       # OpenSimModel: facade generica, show(), composizione di modelli, OpenSimModel.from_step
	_registry.py                   # interno: registro Body->OpenSimModel e iterazione generica di un opensim.Set, condivisi da model.py e operators/
	operators/                     # add_*/remove_* generici per corpi, giunti, forze/muscoli, marker, vincoli, ... (package, un file per categoria)
		__init__.py                 # ri-esporta tutti i nomi pubblici: l'import `from opensim_models import operators` non cambia
		bodies.py                   # add_body/remove_body
		joints.py                   # add_joint/remove_joint e i costruttori nominati (add_free_joint, add_pin_joint, ...)
		attachment.py               # attach_component
		frames.py                   # add_offset_frame
		primitives.py               # add_box_body/add_cylinder_body/add_sphere_body
		forces.py                   # add_force/remove_force, add_muscle/remove_muscle
		markers.py                  # add_marker/remove_marker
		constraints.py              # add_constraint e i costruttori nominati (add_weld_constraint, add_point_constraint, add_coordinate_coupler_constraint, add_point_on_plane_constraint)
		auxiliary.py                # controller/contact-geometry/probe
		contact.py                  # add_contact_sphere/add_contact_half_space/add_contact_mesh (pura geometria) e add_sliding_point_contact (ExponentialContactForce, forza reale alternativa ad add_point_on_plane_constraint)
		rotation.py / translation.py  # rotate_object/translate_object
		geometry.py                 # euclidean_distance (pura geometria) e from_global_to_local/from_local_to_global (leggono la posa corrente di un oggetto)
		_shared.py / _spatial.py    # interno: helper generici condivisi fra i file sopra
	_cad_import.py                 # interno: lettura STEP/STP e generazione mesh per OpenSimModel.from_step
	_primitives.py                 # interno: writer mesh STL per box/cilindro/sfera, usati da operators/primitives.py e da components/box.py
	_geometry.py                   # interno: costruzione di una sorgente VTK (mesh/Brick/Cylinder/Sphere) e dei suoi bounds, condivisa da _gui/visualizer.py e components.Body.corners
	_gui/                          # interno: tutta la finestra interattiva aperta da show() -- mai importato direttamente dai consumer del package
		__init__.py
		visualizer.py               # vista 3D interattiva (VTK), incluso lo stile di navigazione della camera e capture_frame() per l'export
		player.py                   # finestra Tk unificata (vista 3D incorporata + controlli Playback/View/Export) e la logica di riproduzione (MotionData, MotionPlayer)
		tooltip.py                  # il tooltip (angoli stondati) con nome ed (x, y, z) del componente sotto al mouse
		export.py                   # salvataggio della vista corrente in PNG o di una motion caricata in MP4
		win32_embed.py              # interno: incorporamento Win32 della finestra nativa VTK dentro quella Tk (solo Windows)
	models/
		user/
			user.py                    # User(OpenSimModel, _PostureMixin, _JointCenterMixin): scaling antropometrico e posa
			_posture_table.py          # tabella dichiarativa (coordinata, range, docstring) per ogni setter/getter di postura
			_posture_generated.py      # GENERATO da scripts/generate_user_code.py -- non modificare a mano
			_joint_center_table.py     # tabella dichiarativa per ogni property di centro articolare
			_joint_centers_generated.py  # GENERATO da scripts/generate_user_code.py -- non modificare a mano
			_data.py                   # interno: caricamento ANSUR e percentili
			_mapping.py                # interno: mappa ANSUR -> corpi OpenSim
			assets/
				ansur_ref.csv           # riferimenti antropometrici ANSUR II
				rajagopalaiulrich2023.osim  # modello OpenSim base di User
				meshes/*.vtp            # mesh per il rendering di User
	components/
		__init__.py                 # wrapper Python-friendly (Body, Marker, Joint, OffsetFrame, WeldConstraint, PointConstraint, ConstantDistanceConstraint, ExponentialContactForce, ...) dietro gli accessori di OpenSimModel
		box.py                      # Box(components.Body): parallelepipedo rigido generico
		screen.py                   # Screen(components.Body): pannello plexiglass parametrico
		assets/meshes/               # mesh generata automaticamente per Box/Screen, di default (vedi mesh_dir sotto)
scripts/
	generate_user_code.py          # rigenera _posture_generated.py/_joint_centers_generated.py dalle tabelle; non installato col package, solo per chi sviluppa opensim-models
tests/
	test_model.py                  # test esaustivi di OpenSimModel (facade + composizione)
	test_model_wrapper_signatures.py  # verifica che ogni convenience method di OpenSimModel abbia la stessa firma dell'omonima funzione in operators
	test_operators.py              # test esaustivi di opensim_models.operators
	test_operators_public_api.py   # verifica che `operators.__all__`/ogni nome pubblico risolvano esattamente come prima dello split in package
	test_player.py                 # test della logica pura di riproduzione (opensim_models._gui.player)
	test_user.py                   # test esaustivi di User (dati ANSUR, scaling, postura)
	test_user_generated_code_is_fresh.py  # verifica che _posture_generated.py/_joint_centers_generated.py siano allineati alle tabelle
	test_screen.py                 # test esaustivi di Screen (dimensionamento, posa, mesh)
	test_box.py                    # test esaustivi di Box (dimensioni/massa, posa live, spigoli, mesh)
	test_geometry.py               # test di operators.euclidean_distance (nessuna dipendenza da OpenSim) e di from_global_to_local/from_local_to_global (richiedono OpenSim)
	test_cad_import.py             # test di OpenSimModel.from_step (richiede pythonocc-core)
```

Ogni modello specifico (oggi solo `User`) vive nella propria sottocartella sotto `models/`, con il proprio codice e i propri asset; nuovi modelli (es. un attrezzo da palestra completo) si aggiungono allo stesso modo, come ulteriori sottoclassi di `OpenSimModel`. I componenti (`Box`, `Screen`) vivono invece in `components/`, distinti da `models/` proprio perché non sono container: nuovi componenti si aggiungono come ulteriori sottoclassi di `components.Body`, nello stesso file (uno per componente).

I setter/getter di postura e le property di centro articolare di `User` (`set_left_hip_flexionextension`, `left_hip`, ...) non sono scritti a mano: `user.py` eredita da due mixin (`_PostureMixin`, `_JointCenterMixin`) generati come codice sorgente letterale -- non con `setattr`/metaclassi -- a partire dalle tabelle dichiarative `_posture_table.py`/`_joint_center_table.py`, proprio per restare completamente visibili all'autocompletamento/IntelliSense come se fossero scritti a mano. Per aggiungere una nuova coordinata di postura o un nuovo centro articolare: aggiungi una riga alla tabella corrispondente, poi esegui `python scripts/generate_user_code.py` per rigenerare i due file `_*_generated.py` (committati nel repository); `tests/test_user_generated_code_is_fresh.py` fallisce se li dimentichi.

**Superficie pubblica.** L'unico simbolo importabile per ciascun modello/componente è la sua classe (`User`, `Screen`, `Box`): `from opensim_models import OpenSimModel, User, Screen, Box` è l'API pubblica principale del package. Tutto il resto -- dataset ANSUR, funzioni di risoluzione dei percentili, mappe di scaling, lettura CAD, le tabelle/il codice generato di `User` -- è dettaglio implementativo del modello che lo usa: vive in moduli non riesportati dai vari `__init__.py` (per `User`, i moduli con prefisso `_`, come `_data.py` e `_mapping.py`) e non è pensato per essere importato direttamente. Fa eccezione `opensim_models.operators` (vedi sotto): un modulo di utilità pensato per essere importato direttamente (`from opensim_models import operators`), non riesportato al livello superiore del package per restare distinto dalle classi di modello/componente -- è organizzato internamente come un package (un file per categoria di componente), ma questo è un dettaglio implementativo: `opensim_models.operators.<nome>` risolve esattamente come prima.

Il modello di `User` referenzia 85 mesh VTP, tutte incluse nella cartella `models/user/assets/meshes/`. Sono presenti anche quattro alias aggiuntivi per femori e tibie.

## Installazione

Il package richiede Python 3.10 o superiore, NumPy, Pandas e SciPy (quest'ultima usata per l'estrapolazione PCHIP di `User` quando l'altezza richiesta esce dal range ANSUR, vedi sotto). I binding Python di OpenSim sono una dipendenza nativa opzionale, necessaria solo per costruire/scalare/visualizzare un modello effettivo:

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

L'altezza è espressa in centimetri. Quando `height_cm` è presente, ogni misura numerica (statura compresa) viene risolta **direttamente da quell'altezza**, tramite una regressione PCHIP per-misura contro i soggetti ANSUR del sesso indicato, e ha precedenza su `percentile`:

```python
user = User("M", height_cm=175.0)

print(user.height)       # 175.0: esattamente l'altezza richiesta
print(user.percentile)   # percentile empirico corrispondente a 175 cm, solo informativo
```

Il percentile riportato da `user.percentile` è calcolato a parte (percentile empirico della statura nel gruppo ANSUR del sesso indicato) ed è puramente informativo: le misure non passano più da un lookup per percentile, quindi `user.height` coincide esattamente con l'altezza richiesta, non è più una statura ANSUR approssimata.

Un'altezza fuori dal range osservato da ANSUR per quel sesso **non viene più rifiutata**: viene estrapolata (con SciPy `PchipInterpolator`, coda lineare oltre i dati per evitare oscillazioni della cubica) sia per il percentile sia per ogni singola misura, ed emette un `UserWarning`:

```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    tall_user = User("M", height_cm=205.0)  # oltre il massimo ANSUR maschile (~199 cm)

print(caught[0].category)   # UserWarning
print(tall_user.height)     # 205.0
```

### Massa esplicita

Analogamente all'altezza, è possibile passare `mass_kg` al costruttore per richiedere una massa totale specifica:

```python
heavy_user = User("M", height_cm=180.0, mass_kg=110.0)

print(heavy_user.mass_kg)                        # 110.0: esattamente la massa richiesta
print(heavy_user.model.getTotalMass(heavy_user.state))  # 110.0 (a meno di arrotondamento in virgola mobile)
```

Se `mass_kg` non viene passato, la massa non resta un semplice sottoprodotto volumetrico della scalatura geometrica guidata dall'altezza: viene comunque risolta "come le altre misure", cioè dal riferimento ANSUR `weight_kg` risolto per l'altezza/percentile di quell'utente (via `resolve_reference`, la stessa regressione PCHIP contro l'altezza usata per ogni altra misura, oppure il percentile diretto se l'altezza non è stata passata):

```python
default_user = User("M", height_cm=180.0)

print(default_user.mass_kg)                               # peso ANSUR medio risolto per 180 cm
print(default_user.anthropometry.values["weight_kg"])      # stesso valore
print(default_user.model.getTotalMass(default_user.state)) # idem
```

In entrambi i casi la geometria dei segmenti (lunghezze, marker, centri articolari) resta quella determinata dalla sola altezza: `mass_kg` scala **solo** la massa e il tensore d'inerzia di ogni corpo (fattore scalare uniforme, via `opensim.Body.scaleMass`, che scala insieme massa e tensore d'inerzia lasciando invariato il baricentro locale), non le dimensioni. `mass_kg` deve essere un valore finito e strettamente positivo, altrimenti il costruttore solleva `ValueError`.

A differenza di `gender`/`height_cm`/`percentile` (risolti una sola volta alla costruzione, senza alcun setter), la massa totale può essere ricalibrata anche dopo, con `set_mass_kg` -- non ricostruisce il modello da zero, riapplica solo la correzione di massa indipendente sul modello già scalato geometricamente, preservando la postura corrente:

```python
user = User("M", height_cm=180.0)
user.set_mass_kg(130.0)

print(user.mass_kg)                           # 130.0
print(user.model.getTotalMass(user.state))    # 130.0
```

`User` espone solo `mass_kg`/`set_mass_kg` (non anche un alias `mass`/`set_mass`): a differenza di `Box` -- dove `mass`/`mass_kg` coesistono perché `mass` è ereditato dalla classe generica `components.Body` e va comunque reindirizzato verso `set_mass_kg` -- `User` non eredita alcuna property `mass` da ridefinire, quindi un solo nome (quello esplicito sull'unità di misura) evita l'ambiguità.

## Scaling antropometrico

Durante la costruzione di `User` vengono eseguiti questi passaggi:

1. caricamento e normalizzazione del CSV ANSUR;
2. filtraggio per sesso;
3. calcolo del percentile comune per tutte le misure;
4. confronto con il riferimento del 50° percentile dello stesso sesso;
5. costruzione dei fattori `(x, y, z)` per i corpi OpenSim;
6. applicazione tramite `OpenSim.Model.scale` e `ScaleSet` (ereditata da `OpenSimModel.scale_bodies`);
7. ricalibrazione della massa totale del modello (`User._rescale_total_mass`): un fattore scalare uniforme (`mass_kg` richiesto, o altrimenti il `weight_kg` ANSUR risolto, diviso per la massa totale ottenuta dal solo scaling geometrico) viene applicato a ogni corpo con `opensim.Body.scaleMass`, che scala coerentemente massa e tensore d'inerzia. Questo passaggio avviene **sempre**, non solo quando `mass_kg` è esplicito -- vedi "Massa esplicita" sopra.

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
print(user.ground)       # opensim.Ground, equivalente a user.model.getGround()
print(user.bodies)       # dict[str, components.Body], per nome
print(user.joints)       # dict[str, components.Joint]
print(user.muscles)      # dict[str, components.Muscle]
print(user.markers)      # dict[str, components.Marker]
print(user.coordinates)  # dict[str, components.Coordinate]
print(user.frames)       # dict[str, components.OffsetFrame]: vuoto di default, vedi sotto
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

Ognuno di questi setter ha una property di sola lettura omonima (senza `set_`), equivalente a `coordinate_degrees(...)` ma già legata al nome specifico della coordinata:

```python
print(user.left_hip_flexionextension)    # 0.0
user.set_left_hip_flexionextension(25.0)
print(user.left_hip_flexionextension)    # 25.0
print(user.right_knee_flexionextension)
print(user.lumbar_extension)
```

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
        model, model.opensim.FreeJoint("b1_to_ground", model.ground, body)
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

### Ruotare e traslare un oggetto o un intero modello

`rotate_object`/`translate_object` riposizionano qualcosa che esiste **già** (un marker, il frame statico di un giunto, o un intero modello) -- a differenza delle funzioni `add_*`/`remove_*` viste sopra non serve `reinitialize=`/`model.structural_change()`: lo richiamano da sole.

`rotate_object(obj, origin, direction, angle_deg)` ruota `obj` di `angle_deg` gradi attorno all'asse passante per `origin` e diretto come `direction` (non serve che sia un versore):

```python
from opensim_models import operators

# ruota l'intero modello di 90° attorno all'asse Z passante per l'origine
user.rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)

# equivalente, chiamando direttamente operators.rotate_object
operators.rotate_object(user, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)
```

Su un intero `OpenSimModel` (o `User`/`Screen`), ruota ogni giunto di quel modello collegato direttamente al ground (per `User`, `ground_pelvis`) della stessa quantità: l'intero corpo ruota rigidamente, senza toccare nessun angolo articolare relativo (`hip_flexion_r`, `knee_angle_r`, ...) -- utile per riorientare un intero modello (es. per appoggiarlo contro un piano di riferimento) senza alterarne la postura interna.

Su un singolo componente, invece, cambia in base al tipo:

- `opensim.Marker`: aggiorna la sua `location` (nel proprio parent frame) -- un marker non ha orientamento, quindi cambia solo la posizione;
- `opensim.PhysicalOffsetFrame` (es. il parent/child frame di un giunto costruito con `add_weld_joint`/`add_slider_joint`/...): aggiorna sia `translation` che `orientation` -- è così che si può ri-orientare, dopo la costruzione, il posizionamento statico di un giunto impostato inizialmente con `position=`/`orientation_deg=`;
- qualunque altra cosa accettata (un `Body`, un `Joint`, un `Frame` generico, o una semplice coordinata): non esiste un modo generico per *spostare* questi oggetti (la posizione di un `Body` dipende interamente dal giunto che lo collega, un `Joint` non è di per sé un oggetto posizionabile), quindi la funzione calcola e restituisce la posizione ruotata senza modificare nulla -- utile per calcolare un argomento `position=`/`origin=` per un'altra chiamata (es. costruire un nuovo giunto con `add_weld_joint`) senza effetti collaterali.

`origin` (il perno) può essere una coordinata `(x, y, z)` oppure un qualunque componente con una posizione nel ground frame (un `Marker`, un `Joint`, un `Frame`). Se né `obj` né `origin` sono componenti di un modello (sono entrambi semplici coordinate), `rotate_object` funziona anche senza alcun `OpenSimModel`:

```python
operators.rotate_object((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)  # -> (0.0, 1.0, 0.0)
```

`translate_object(obj, direction)` è l'equivalente per una traslazione pura: non serve un perno (ogni punto di un corpo rigido trasla della stessa quantità), solo lo spostamento `(dx, dy, dz)`, in metri, nel ground frame. Stessa casistica di `rotate_object` per cosa viene effettivamente modificato (modello intero → ogni giunto agganciato al ground; `Marker`/`PhysicalOffsetFrame` → mutati in place; tutto il resto → sola lettura), con la differenza che una traslazione non tocca mai l'orientamento:

```python
user.translate((1.0, 0.0, 0.0))  # sposta l'intero User di 1 m lungo X, stessa postura
```

Entrambe le funzioni accettano `inplace` (default `True`): a `False`, l'oggetto passato **non** viene toccato -- al suo posto viene clonato il modello a cui appartiene (`OpenSimModel.copy()`), la rotazione/traslazione viene applicata alla copia, e viene restituito **l'oggetto modificato dentro quella copia** (non più una semplice posizione):

```python
marker_ruotato = operators.rotate_object(
    marker, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0, inplace=False
)
marker.get_location()           # invariato: il marker originale non è stato toccato
marker_ruotato.get_location()   # la posizione ruotata, su un marker che vive in un modello clonato
```

Se `obj` è un intero modello, la copia restituita **è** il modello stesso (ruotato/traslato), non solo un componente al suo interno -- vale anche per `model.rotate(..., inplace=False)`/`model.translate(..., inplace=False)`, che restituiscono una copia indipendente del modello invece della posizione. Sui casi di sola lettura (`Body`, `Joint`, `Frame` generico, coordinata semplice) `inplace` non ha alcun effetto: non c'è mai nulla da modificare, quindi viene sempre restituita la sola posizione calcolata.

`model.rotate(origin, direction, angle_deg, inplace=True)` e `model.translate(direction, inplace=True)` (ereditati da `OpenSimModel`, usati sopra) sono scorciatoie equivalenti a chiamare `operators.rotate_object`/`operators.translate_object` passando il modello stesso come primo argomento -- comode quando si lavora già con un'istanza di modello e non si vuole importare `operators` esplicitamente.

Il `PhysicalOffsetFrame` di un giunto menzionato sopra è ora anche accessibile direttamente, già wrappato, senza passare da `rotate_object`/`translate_object`: `joint.parent_frame`/`joint.child_frame` restituiscono un `components.OffsetFrame` (quando il frame è davvero un `PhysicalOffsetFrame`, sempre vero per i giunti di questo pacchetto e per quelli del modello Rajagopal bundlato), con `translation`/`set_translation`, `orientation_deg`/`set_orientation_deg` e la coppia `position_global`/`position_local` (quest'ultima uguale a `translation`) già viste per `Body`/`Marker`/`Joint`/`ContactGeometry`:

```python
child_frame = user.joint("hip_r").child_frame
print(child_frame.translation)       # offset locale rispetto al femore
print(child_frame.position_global)   # stesso punto, nel ground frame
```

**Attenzione**: se questo frame è il parent/child frame di un giunto (non, ad esempio, il frame di un `WeldConstraint`), `set_translation`/`set_orientation_deg` spostano il **corpo** a cui il frame appartiene, non il frame stesso nel ground frame -- un giunto esiste apposta per far coincidere i suoi due frame, quindi `position_global` di questo frame resta invariato dopo la modifica (confermato direttamente). Per spostare un punto a un target preciso nel ground frame usa `rotate_object`/`translate_object`, che risolvono per te il valore locale corretto invece di limitarsi a scrivere quello richiesto.

### Convertire un punto fra ground frame e il frame locale di un oggetto

`from_global_to_local(coordinates, obj)`/`from_local_to_global(coordinates, obj)` convertono un punto `(x, y, z)` qualunque (non necessariamente l'origine di `obj`) fra il ground frame e il frame locale di `obj`, leggendo la posa corrente di `obj` (`position_global`/orientamento) -- sono l'una l'inversa dell'altra:

```python
femore = user.body("femur_r")

punto_locale = operators.from_global_to_local((0.0, 1.0, 0.0), femore)
punto_ground = operators.from_local_to_global(punto_locale, femore)  # torna (0.0, 1.0, 0.0)
```

`obj` accetta sia il wrapper di questo pacchetto (`Body`, `Box`, `Screen`, `OffsetFrame`, ...) sia il rispettivo oggetto `opensim` grezzo, purché appartenga già a un `OpenSimModel`/`User` vivo -- stessa ampiezza accettata da `position_global`/`position_local` sui wrapper in `components`. Un `Marker` o un `Joint` non sono di per sé frame orientabili (un marker non ha un proprio orientamento, un giunto ne ha due -- parent e child): passare invece il parent frame del marker (`marker.parents[0]`) o `joint.parent_frame`/`joint.child_frame`.

A differenza di ogni `add_*`/`remove_*` visto sopra, queste due funzioni non prendono né `model=` né `reinitialize=`: non modificano nulla, e la posa di `obj` viene letta -- tramite lo stesso helper condiviso (`components._position_and_rotation_in_ground`) dietro ogni `position_global`/`position_local` di questo pacchetto -- dal modello a cui `obj` già appartiene, risolto automaticamente nello stesso modo di `rotate_object`/`translate_object` per un oggetto grezzo.

### Creare un offset frame autonomo (`add_offset_frame`)

`add_offset_frame(model, name, body, translation=..., orientation_deg=..., *, reinitialize=False)` costruisce un nuovo `opensim.PhysicalOffsetFrame` -- un punto/orientamento nominato, solidale a `body` -- e lo aggiunge al modello in una sola chiamata, restituendolo già wrappato in `components.OffsetFrame` (stesso wrapper usato per `joint.parent_frame`/`child_frame`, ma qui è il *costruttore* mancante: prima esisteva solo un offset frame interno, anonimo, usato da `add_model` per il proprio bookkeeping). Equivalente anche come metodo su `OpenSimModel` (quindi ereditato da `User`):

```python
elbow_attachment = user.add_offset_frame(
    "elbow_pad_point", user.body("humerus_r"), translation=(0.0, -0.05, 0.03)
)
print(elbow_attachment.position_global, elbow_attachment.parents)   # (humerus_r,)
```

**Dove viene agganciato, e perché, verificato direttamente**: come sotto-componente del body stesso (`body.addComponent(frame)`, path `/bodyset/<body>/<name>`), non alla radice del modello come fa l'ancora interna anonima di `add_model` (quella è dichiaratamente "non tracciata per la rimozione... orfana dopo `remove_model`"). Confermato che agganciarlo al body è la scelta che sopravvive alla fusione di modelli (`model + other`): clonare un body (come fa ciascuna delle 8 categorie di `_MERGE_SETS`) clona anche questo suo sotto-componente, ritrovabile allo stesso path nel modello risultante; un componente alla radice ne resterebbe invece fuori, come l'ancora interna. `copy()` (un clone completo del modello) lo preserva in entrambi i casi.

**Riusabile ovunque il pacchetto accetta un body/frame**, confermato direttamente: come `body=` in `add_marker`/`add_contact_sphere`/etc. (è un `opensim.PhysicalFrame` a tutti gli effetti), e come `to=` in `attach_component` -- che però di norma userebbe il centro di massa (`parent_point="com"`, il default) come punto di aggancio: dato che un offset frame non ha massa propria, questo caso ora ricade sull'origine del frame stesso (`(0, 0, 0)`), non più un `AttributeError`:

```python
new_body_joint = operators.attach_component(model, new_body, to=elbow_attachment.raw)
```

**`model.frames`** elenca i frame autonomi così creati, sullo stesso modello di `.bodies`/`.muscles`/`.constraints`. OpenSim non ha un `FrameSet`: l'unico punto d'accesso nativo, `getFrameList()`, restituisce *ogni* componente di tipo `Frame` nell'albero del modello -- il che include anche ogni `Body`/`Ground` (anche loro `Frame` a tutti gli effetti, già esposti da `.bodies`/`.ground`) e i due `PhysicalOffsetFrame` che ogni giunto/`WeldConstraint` possiede per il proprio aggancio parent/child (già raggiungibili via `joint.parent_frame`/`child_frame` o `weld.frame1`/`frame2`). Verificato empiricamente sul modello base (`User`, 22 body/22 giunti): `getFrameList()` restituisce 67 elementi, e **tutti** ricadono in una di queste categorie già coperte altrove. Per comportarsi come `.bodies`/`.muscles` -- una categoria nuova, non ridondante, vuota finché non ci si aggiunge esplicitamente qualcosa -- `model.frames` esclude quindi `Body`/`Ground` e ogni frame il cui proprietario (`getOwner()`) è un `Joint` o un `Constraint`; quel che resta sono i frame autonomi creati con `add_offset_frame` (dispatch automatico a `components.OffsetFrame`, stesso meccanismo di `.constraints`/`.contact_geometries`):

```python
print(model.frames)                     # {} su un modello senza offset frame autonomi
print(model.frames["elbow_pad_point"])  # components.OffsetFrame, dopo l'add_offset_frame sopra
```

### Vincoli e contatto con punti dedicati

`add_weld_constraint`, `add_point_constraint` e `add_point_on_plane_constraint` restituiscono ora, rispettivamente, un `components.WeldConstraint`, `components.PointConstraint` o `components.ConstantDistanceConstraint` (non più il generico `Constraint`) -- idem per `model.constraints`, che dispatcha automaticamente al tipo più specifico disponibile per ogni vincolo già nel modello. Ciascuno espone i propri punti, nel ground frame e nel frame locale:

```python
weld = operators.add_weld_constraint(model, "w1", body1, body2, position1=(0.1, 0, 0))
print(weld.point1_global, weld.point1_local)   # punto su body1
print(weld.point2_global, weld.point2_local)   # punto su body2
print(weld.frame1, weld.frame2)                # ciascuno un components.OffsetFrame

point = operators.add_point_constraint(model, "p1", ground, (0, 0, 0), body, (0.1, 0, 0))
print(point.point1_global, point.point2_global)

plane = operators.add_point_on_plane_constraint(model, "pl1", body, pt, plane_body, plane_pt, normal)
print(plane.point1_global)   # il punto vincolato
print(plane.point2_global)   # l'ancora calcolata, NON plane_pt -- vedi la sua docstring
print(plane.distance)
```

Un `opensim.CoordinateCouplerConstraint` (costruito con `add_coordinate_coupler_constraint`) non ha un punto spaziale singolo: resta wrappato nel generico `Constraint`, invariato.

Analogamente, `add_sliding_point_contact` restituisce ora un `components.ExponentialContactForce`, con `point_global` (il punto di contatto, la "station" nativa di OpenSim) e `plane_point_global` (un punto sul piano) -- **senza** l'equivalente locale per nessuno dei due: confermato direttamente che questa forza non espone i propri riferimenti a corpo/frame come `Socket` OpenSim (`getSocketNames()` è vuoto) né una `getStation()` utilizzabile dai binding Python di questa installazione, quindi non c'è modo di risalire al corpo/frame originale da questo solo oggetto.

### Risolvere condizioni cinematiche senza l'`assemble()` nativo (`solve_coordinates`, `solve_point_coincidence`)

`Model.assemble()` -- e, più in generale, qualunque percorso che si appoggi allo stesso `SimTK::Assembler` nativo (es. lasciar risolvere a `initSystem()` un `Constraint` appena aggiunto e non ancora soddisfatto, o `Coordinate.setValue(state, value, enforce_constraints=True)` su una coordinata coinvolta in un vincolo) -- è confermato, con un minimal repro diretto, **crashare il processo nativamente** (non un'eccezione catturabile) su questo modello `User` ogni volta che c'è un vincolo realmente non soddisfatto da risolvere con coordinate libere disponibili. È lo stesso tipo di limite già documentato altrove in questo pacchetto: la nota su `CoordinateCouplerConstraint` in "Architettura: Model, State e propagazione" sopra, il docstring di `add_point_on_plane_constraint` (un `ConstantDistanceConstraint` non soddisfatto fra due corpi entrambi discendenti dal ground crasha `initSystem()`), e quello di `settle_under_gravity` (stesso genere di crash nativo, altro percorso).

Due operazioni sono invece confermate sempre sicure su questo modello: `Coordinate.setValue(state, value, enforce_constraints=False)` (scrive il valore grezzo, non tocca mai l'assembler) e `Model.realizePosition(state)` (propaga quel valore a ogni quantità derivata -- posizioni di corpi/marker/giunti -- senza risolvere alcun vincolo). `solve_coordinates` costruisce un risolutore numerico interamente sopra queste due primitive più `scipy.optimize.least_squares`: qualunque condizione cinematica andrebbe altrimenti affidata all'assembler nativo (es. "questo punto deve stare su quel piano", "questi due punti devono coincidere") viene invece espressa come una funzione residuo Python, azzerata iterando candidati sulle coordinate tramite la coppia sicura `set_value`/`realizePosition`:

```python
def residual(x):
    return [user.right_shoulder[0] - target_x]  # x già impostato su model.state

result = user.solve_coordinates(residual, ["lumbar_extension"], x0=[0.0])
print(result.success, result.x)   # risultato di scipy.optimize.least_squares
```

`solve_point_coincidence` è una convenienza pronta all'uso, costruita sopra `solve_coordinates`, per il caso di gran lunga più ricorrente in questo tipo di studio ergonomico: far coincidere uno o più punti solidali al corpo (una spalla, un tallone, un punto su bacino/tronco) con un punto fisso su un componente della macchina (un pad, uno schienale, un poggiapiedi) -- senza saldare i due corpi con un vincolo nativo, quindi senza correre il rischio di crash sopra per un vincolo non soddisfatto al momento della costruzione:

```python
# forma 1: due getter a zero argomenti, per punti qualunque (centro
# articolare, marker, punto proiettato, ...)
result = user.solve_point_coincidence(
    [(lambda: user.right_shoulder, lambda: target_point)],
    ["lumbar_extension"],
)

# forma 2: (frame_a, punto_a, frame_b, punto_b) -- un punto fisso nel frame
# locale di ciascun corpo, tipicamente il caso spallaccio/pad o tallone/
# poggiapiedi
result = user.solve_point_coincidence(
    [(user.bodies["calcn_r"], heel_point_local, footrest, footrest_point_local)],
    ["hip_flexion_r", "knee_angle_r", "ankle_angle_r"],
)
```

Le due forme sono miscelabili liberamente nella stessa chiamata; la seconda riusa lo stesso `from_local_to_global` già visto sopra, invece di ricavare la trasformazione a mano. Entrambe le funzioni restituiscono l'`OptimizeResult` di `scipy.optimize.least_squares` così com'è (`x`, `fun`, `cost`, `success`, `status`, `message`, ...), e per default (`raise_on_failure=True`) sollevano un `RuntimeError` -- con lo stesso `status`/`message` di scipy -- se la convergenza non viene raggiunta, lasciando comunque il modello al miglior `x` trovato; passa `raise_on_failure=False` per ispezionare direttamente il risultato invece di ricevere l'eccezione. In ogni caso, un fallimento qui è **sempre** un'ordinaria non convergenza numerica, mai il crash nativo descritto sopra: nessuna delle due funzioni chiama mai `Model.assemble()`, `Manager`, o `setValue` con `enforce_constraints=True`.

### Geometrie di contatto nominate (`ContactSphere`, `ContactHalfSpace`, `ContactMesh`)

`add_contact_sphere`, `add_contact_half_space` e `add_contact_mesh` costruiscono la rispettiva `opensim.ContactGeometry` e la agganciano al modello in una sola chiamata -- stesso pattern "costruisci e aggiungi" di `add_box_body`/`add_weld_constraint`/etc. -- invece di dover costruire l'oggetto OpenSim a mano con SWIG grezzo e passarlo al generico `add_contact_geometry`. Equivalenti anche come metodo su `OpenSimModel` (quindi ereditati da `User`): `user.add_contact_sphere(...)`, ecc. `model.contact_geometries`/`add_contact_geometry` dispatchano automaticamente al wrapper più specifico, stesso meccanismo di `Constraint`/`Force`:

```python
sphere = user.add_contact_sphere("cs1", femur, radius=0.04, location=(0, -0.1, 0))
print(sphere.radius, sphere.position_global, sphere.parents)   # (femur,)

half_space = user.add_contact_half_space("chs1", backrest_body, orientation_deg=(0, 0, 90))

mesh = user.add_contact_mesh("cm1", pelvis, "pelvis_contact.obj")
print(mesh.filename)
```

Tutti e tre ereditano da `ContactGeometry` `location`/`set_location`, `orientation_deg`/`set_orientation_deg` (offset/orientamento nel frame a cui sono agganciati) e `position_global`/`position_local`/`parents`; `ContactSphere` aggiunge `radius`/`set_radius` (validato, finito e strettamente positivo), `ContactMesh` aggiunge `filename`/`set_filename`, `ContactHalfSpace` non aggiunge nulla (il semispazio solido è il lato locale +X, quindi quello apribile/di contatto è -X -- convenzione nativa di OpenSim, non una scelta di questo pacchetto).

**Attenzione, lo stesso limite già documentato per `add_sliding_point_contact` non è sparito**: queste `ContactGeometry` sono pura geometria -- aggiungerle non crea di per sé alcuna fisica di contatto. Se l'obiettivo è appaiarle a `HuntCrossleyForce`/`ElasticFoundationForce` per un contatto a forza "classico", quella strada resta un vicolo cieco in questa installazione (la classe annidata `ContactParameters` di cui hanno bisogno non è raggiungibile dai binding Python, e i setter "appiattiti" che sembrano un sostituto causano un crash nativo allo stage delle accelerazioni -- vedi il docstring di `add_sliding_point_contact`, "Why `ExponentialContactForce`, not `HuntCrossleyForce`"). `ExponentialContactForce` resta l'unico contatto a forza funzionante in questa installazione.

**Un secondo problema nativo confermato e aggirato da `add_contact_mesh`**: il costruttore "tutto in una volta" di `opensim.ContactMesh` (`ContactMesh(filename, location, orientation, frame, name)`) causa un crash nativo del processo (non un'eccezione catturabile) in questa installazione, riprodotto sia con un `.stl` scritto da questo stesso pacchetto sia con un `.vtp` bundlato con `User` -- `add_contact_mesh` costruisce l'oggetto pezzo per pezzo (costruttore vuoto + `set_filename`/`set_location`/`set_orientation`/`setName`/`connectSocket_frame`), confermato funzionare correttamente, e lo fa per te.

### Relazioni tra componenti (`parents`)

Ogni wrapper per cui "da cosa dipende"/"a cosa è collegato" ha un senso espone anche una property `parents` (una tupla, mai una lista, come `Box.corners`), nei due versi:

- **in avanti** (il componente conosce già il proprio riferimento): `Marker.parents` -- il body a cui è agganciato (1 elemento); `Joint.parents` -- `(parent_body, child_body)`, risolti fino al corpo/ground reale anche quando il frame è un `OffsetFrame` intermedio; `Muscle.parents` -- ogni body distinto attraversato dal percorso (via `getGeometryPath().getPathPointSet()`, duplicati rimossi); `ContactGeometry.parents` -- il body a cui è attaccata; `WeldConstraint`/`PointConstraint`/`ConstantDistanceConstraint.parents` -- `(body1, body2)`; `OffsetFrame.parents` -- il body/ground del proprio frame genitore.
- **all'indietro** (nessun riferimento nativo, va cercato): `Body.parents` -- ogni `Joint`/`Muscle`/vincolo-con-wrapper-dedicato/`Marker` del modello che referenzia quel body, trovato scansionando `model.joints`/`model.muscles`/`model.constraints`/`model.markers` e controllando quali elencano quel body nel proprio `parents` (in avanti). **Costo**: a differenza di tutte le altre property di questo modulo (che leggono un singolo valore), `Body.parents` visita ogni componente di quelle quattro categorie ad ogni lettura, senza cache -- evita di chiamarlo in un ciclo stretto su molti body.

```python
femur = user.body("femur_r")
for parent in femur.parents:
    print(type(parent).__name__, parent.name)   # Joint hip_r, Joint walker_knee_r, Muscle glmax1_r, ...
```

Non tutti i tipi concreti hanno un `parents`: un `opensim.CoordinateCouplerConstraint` (nessun body/frame referenziato, solo coordinate) resta sul generico `Constraint`, senza `parents`; `ExponentialContactForce` non lo espone per lo stesso motivo documentato sopra per `plane_point_local` (nessun riferimento a corpo/frame recuperabile dai binding). `Body.parents` salta entrambi i casi di conseguenza.

## Dati ANSUR risolti

Il caricamento del CSV ANSUR e il calcolo dei percentili (`load_ansur`, `resolve_reference`) sono dettagli implementativi interni di `User`, non parte della superficie pubblica del package (vedi sopra). I valori risolti restano comunque accessibili dopo aver costruito un `User`, tramite la proprietà `anthropometry`:

```python
user = User("F", percentile=75.0)

reference = user.anthropometry
print(reference.height_cm)
print(reference.values["footlength"])
```

Le misure lineari ANSUR restano nelle unità sorgente, prevalentemente millimetri. `stature_m` è in metri; `height_cm` è una proprietà di comodo in centimetri.

## Centri articolari e misure derivate

Oltre ad `anthropometry`, `User` espone una serie di proprietà pubbliche di comodo, tutte in metri.

**Centri articolari** -- posizione `(x, y, z)` nel ground frame di ogni giunto del modello, più il dizionario `joint_centers` con tutti insieme:

```python
print(user.left_hip)        # (x, y, z) in metri
print(user.right_ankle)
print(user.joint_centers)   # {"pelvis": (...), "left_hip": (...), ...}
```

Sono disponibili: `pelvis`, `left_hip`/`right_hip`, `left_knee`/`right_knee`, `left_patella`/`right_patella`, `left_ankle`/`right_ankle`, `left_subtalar`/`right_subtalar`, `left_mtp`/`right_mtp`, `torso`, `left_shoulder`/`right_shoulder`, `left_elbow`/`right_elbow`, `left_radioulnar`/`right_radioulnar`, `left_wrist`/`right_wrist`.

Queste proprietà riflettono la **postura corrente** del modello (sono calcolate da `getPositionInGround`, con un `realizePosition` automatico, quindi funzionano anche subito dopo un setter di postura, senza dover chiamare `update_state()` prima). Se cambi una coordinata, il valore letto rispecchia la nuova posa:

```python
user.set_right_knee_flexionextension(45.0)
print(user.right_knee)  # posizione aggiornata, nessun update_state() necessario
```

**Centro di massa e sua proiezione a terra**: `com` (centro di massa dell'intero modello nel ground frame) e `cop` (proiezione verticale di `com` sul piano a terra, `y = 0`) -- `cop` qui è una semplice proiezione cinematica di `com`, non un centro di pressione dinamico calcolato dalle forze di contatto:

```python
print(user.com)  # (x, y, z) del baricentro
print(user.cop)  # (x, 0.0, z): stessa x/z di com, proiettato a terra
```

**Traslare il modello**: `set_position(reference, x, y, z)` sposta rigidamente l'intero `User` (attraverso le coordinate di traslazione del bacino, `pelvis_tx`/`pelvis_ty`/`pelvis_tz`) in modo che il punto `reference` -- qualunque posizione nel ground frame già nota, es. `com`, `cop`, o una qualunque entry di `joint_centers`/`left_ankle`/... -- finisca esattamente in `(x, y, z)`. La postura relativa (angoli delle articolazioni) non cambia, chiama internamente `update_state()`:

```python
user.set_position(user.cop, 0.0, 0.0, 0.0)  # centra la proiezione a terra del CoM nell'origine
user.set_position(user.left_ankle, 1.0, 0.0, 0.3)  # porta la caviglia sinistra in un punto preciso
```

**Lunghezze geometriche** -- distanza fra due centri articolari, quindi anch'esse dipendenti dalla postura corrente: `left_arm_length`/`right_arm_length` (spalla-gomito), `left_forearm_length`/`right_forearm_length` (gomito-polso), `left_thigh_length`/`right_thigh_length` (anca-ginocchio), `left_shank_length`/`right_shank_length` (ginocchio-caviglia), `torso_height` (centro anche - centro spalle), `shoulder_width` (centro spalla sinistra - centro spalla destra).

**Misure dirette da ANSUR** -- nessun centro articolare affidabile nel modello scheletrico per queste dimensioni, quindi vengono lette dal valore ANSUR già risolto (`anthropometry.values`), indipendenti dalla postura:

- lunghezze: `left_foot_length`/`right_foot_length`, `left_palm_length`/`right_palm_length` (solo palmo, non le dita);
- l'altezza da terra del piede è approssimata dall'altezza del centro della caviglia: `left_foot_height`/`right_foot_height` (questa sì dipende dalla postura);
- `biacromial_breadth` -- distanza bi-acromiale ANSUR, da confrontare con `shoulder_width` (geometrica, vedi sopra: le due non coincidono, perché la spaziatura dei giunti di spalla sul corpo `torso` non è scalata esplicitamente sulla larghezza bi-acromiale, ma con il rapporto di statura generico, vedi "Scaling antropometrico");
- circonferenze: `left_arm_circumference`/`right_arm_circumference` (bicipite flesso), `left_forearm_circumference`/`right_forearm_circumference` (avambraccio flesso), `neck_circumference`, `chest_circumference`, `waist_circumference`, `hip_circumference` (glutei), `left_thigh_circumference`/`right_thigh_circumference`, `left_calf_circumference`/`right_calf_circumference`;
- profondità sagittali misurate direttamente da ANSUR: `chest_depth`, `waist_depth`, `hip_depth` (glutei);
- larghezze (breadth) misurate direttamente da ANSUR: `chest_width`, `waist_width`, `hip_width` (glutei).

**Profondità e larghezze stimate** -- coscia e polpaccio non hanno né profondità né larghezza ANSUR con cui fittare un'ellisse: `left_thigh_depth`/`right_thigh_depth`/`left_thigh_width`/`right_thigh_width` e i rispettivi `*_calf_*` assumono quindi una sezione circolare (`diametro = circonferenza / π`), un'approssimazione, non una misura ANSUR diretta -- per ciascun lato, `*_width` e `*_depth` coincidono (un cerchio ha un solo diametro).

Non sono disponibili le lunghezze delle singole dita della mano: né ANSUR II né il modello OpenSim Rajagopal-Lai-Uhlrich (il corpo `hand` è un unico rigido, senza giunti per le falangi) contengono questo dato.

## Rendering

Ogni modello registra automaticamente la propria cartella di mesh: per `User` non serve indicare alcun percorso di geometria.

```python
user = User("M")
user.show()
```

`show()` (ereditato da `OpenSimModel`) apre una finestra Tk interattiva con una vista 3D **propria**, basata su VTK -- non il visualizzatore nativo Simbody: quest'ultimo gira come processo separato e non espone a Python alcuna API di posizione del mouse, trasformazione della camera o picking, impedendo di costruire un tooltip con le coordinate reali sotto il cursore. Per aggiungere ulteriori cartelle di geometria (ad esempio per un modello composto, vedi sotto) si può passare `geometry_path` esplicitamente, oppure registrarle in anticipo con `add_geometry_directory(...)`.

La finestra contiene, dall'alto in basso: la vista 3D (su Windows incorporata direttamente nella finestra; su altre piattaforme si apre come finestra separata, perché l'incorporamento usa una chiamata di reparenting specifica di Win32) e tre gruppi di controlli -- Playback, View, Export (vedi sotto). Passando il mouse su un punto del modello compare un tooltip con angoli arrotondati, posizionato accanto al cursore, strutturato come:

```text
<modello>-<componente>
X: ...
Y: ...
Z: ...
```

in coordinate reali OpenSim/ground frame, in metri.

**Navigazione della camera** nella vista 3D:

- trascinare col pulsante sinistro del mouse ruota la vista;
- tenere premuto **Ctrl** mentre si trascina col pulsante sinistro trasla la vista (pan), senza ruotarla;
- la rotellina del mouse zooma/de-zooma;
- un semplice click, senza trascinare, non ha alcun effetto.

**Playback** -- riproduzione di una `motion` caricata (vedi sotto):

- `⏪` / `▶`-`⏸` / `⏹` / `⏩` -- indietro veloce, play/pausa, stop, avanti veloce, con velocità crescente ad ogni click su avanti/indietro (`1x -> 2x -> 4x -> 8x`, nei due versi);
- `Cycle` -- fa ripartire la riproduzione da capo al termine invece di fermarsi;
- uno slider trascinabile per spostarsi rapidamente in un punto qualunque della simulazione.

**View** -- sempre attivo, indipendentemente da una `motion` caricata:

- `Ground` -- mostra/nasconde il piano di riferimento a terra;
- `Muscles` -- mostra/nasconde i percorsi muscolari;
- `Markers` -- mostra/nasconde i marker;
- `Axes` -- mostra/nasconde la terna di assi X/Y/Z (rosso/verde/blu) all'origine del ground frame, utile per leggere a colpo d'occhio l'orientamento del sistema di riferimento;
- `Camera:` -- un menu a tendina con le viste preimpostate (`Front`, `Back`, `Left`, `Right`, `Top`, `Bottom`), che inquadra il modello da quella direzione.

**Export** -- salva su disco, chiedendo sempre posizione e nome file tramite una finestra di selezione file:

- 📷 -- salva la vista 3D corrente come immagine PNG, a 300 dpi;
- 🎬 -- salva l'intera `motion` caricata come video MP4, campionato a `fps` fotogrammi al secondo (vedi sotto); disabilitato quando non è stata passata alcuna `motion`, come i controlli di Playback.

I controlli di Playback e il pulsante Export di salvataggio animazione sono disabilitati -- non nascosti -- quando non è stata passata alcuna `motion`; View e il pulsante Export di salvataggio immagine restano sempre attivi, perché agiscono sulla vista 3D stessa, non su una riproduzione caricata.

### Riprodurre una simulazione (`show(motion=...)`)

Passando `motion` a `show()` -- un file `.mot`/`.sto`, o una `opensim.TimeSeriesTable` già in memoria (es. quella scritta da un'analisi precedente) -- i controlli di Playback descritti sopra diventano attivi:

```python
user.show(motion="simulazione.mot", loop=True, fps=30)
```

Le colonne del file/tabella sono interpretate come coordinate OpenSim per nome (es. `"hip_flexion_r"`); quelle angolari vengono convertite automaticamente da gradi a radianti se la tabella dichiara `inDegrees=yes` (come fanno i file `.mot` standard), quelle traslazionali restano sempre in metri. `loop` imposta lo stato iniziale dell'interruttore "Cycle"; `fps` è la frequenza di aggiornamento della riproduzione e del ridisegno della vista 3D, ed è anche il frame rate usato per campionare ed esportare l'animazione con il pulsante 🎬.

Questo era, in una versione precedente del package, un design a due finestre separate (il visualizzatore nativo Simbody più una finestra Tk agganciata sotto di esso via API Win32): necessario perché i widget interattivi nativi di Simbody (`Visualizer.addSlider`/`addMenu`/`setWindowTitle` -- qualunque cosa accetti una `SimTK::String`) non sono richiamabili da Python in almeno alcune build correnti di OpenSim. Il visualizzatore non è più quello nativo e gira nello stesso processo, quindi la sua finestra viene invece incorporata (solo su Windows) direttamente in un frame di questa stessa finestra Tk: un'unica finestra, nessun aggancio fra finestre separate da mantenere sincronizzato.

La riproduzione gira su un thread Tk dedicato in background: `show(motion=...)` ritorna subito, e solo quel thread deve toccare lo stato del modello finché la finestra resta aperta (`opensim.State`/`opensim.Model` non sono thread-safe). `user.player` espone la macchina a stati della riproduzione (`opensim_models._gui.player.MotionPlayer`) dopo l'ultima chiamata con `motion`, utile per pilotarla/ispezionarla da codice. `user.visualizer` espone invece la vista 3D stessa (`opensim_models._gui.visualizer.VTKVisualizer`), utile per pilotare programmaticamente le visibilità/viste sopra (`set_ground_visible`, `set_muscles_visible`, `set_markers_visible`, `set_axes_visible`, `set_view`) o per salvare un'immagine (`capture_frame()`, vedi `opensim_models._gui.export`) senza passare dai pulsanti.

Il file esportato con `user.export(...)` contiene il modello scalato con la postura corrente. `export()` copia inoltre automaticamente ogni mesh referenziata dai corpi del modello in una cartella `Geometry/` accanto al file `.osim` esportato (la convenzione di nome che OpenSim/Simbody cercano automaticamente accanto a un modello), così l'esportazione è portabile anche senza le cartelle di geometria originali (`models/user/assets/meshes/` per `User`, la cartella di `from_step` per un modello CAD).

## Creare uno schermo (Screen)

`Screen` è un pannello rigido in plexiglass, spesso 1 mm, pensato per rappresentare un monitor/schermo nella scena. È un **componente** (sottoclasse di `components.Body`, non di `OpenSimModel`): non ha un proprio `show()`/`export()` -- va prima aggiunto a un container (vedi "Comporre più modelli") per poter essere visualizzato:

```python
from opensim_models import Screen

screen = Screen()  # 22", 16:9, centrato nell'origine, verticale (angle_deg=90)

print(screen.width_mm, screen.height_mm)  # None, None: dimensione derivata dalla diagonale

(user_model + screen).show()  # un componente da solo non si vede: va aggiunto a un container
```

Le dimensioni si ottengono in due modi alternativi, con priorità automatica: se `width_mm` e `height_mm` sono *entrambi* impostati vincono loro; altrimenti (compreso il default, con entrambi `None`) la dimensione viene calcolata dalla diagonale in pollici (`inches`, default `22`) e dal rapporto di forma (`ratio`, default `"16:9"`):

```python
screen_esplicito = Screen(width_mm=600.0, height_mm=340.0)
screen_diagonale = Screen(inches=27.0, ratio="21:9")
```

`center_x`/`center_y`/`center_z` posizionano il centro del pannello nel sistema di riferimento del ground (metri); `angle_deg` ne definisce l'inclinazione rispetto al ground: `0` disteso a terra, `90` (default) verticale, come un monitor appoggiato su un piano orizzontale.

Ogni parametro del costruttore ha una property in lettura (`width_mm`, `height_mm`, `inches`, `ratio`, `center_x`, `center_y`, `center_z`, `angle_deg`, `mesh_dir`) e un setter dedicato (`set_width_mm`, `set_height_mm`, `set_inches`, `set_ratio`, `set_center_x`, `set_center_y`, `set_center_z`, `set_angle_deg`, `set_mesh_dir`). Ogni setter ricostruisce il corpo OpenSim, la mesh e il giunto verso ground con i parametri aggiornati. Il costruttore accetta anche `name` (default `None`, cioè `"screen_panel"`, anche impostabile dopo con `set_name`): sticky attraverso rebuild/`copy()` esattamente come per `Box` (vedi sotto), determina anche il nome del file mesh (`f"{name}.stl"`) così da non collidere con quello di un altro `Screen`/`Box` nella stessa `mesh_dir`:

```python
screen.set_angle_deg(0)       # ora disteso sul piano orizzontale
screen.set_width_mm(600.0)
screen.set_height_mm(340.0)   # passa in modalità dimensioni esplicite solo una volta impostate entrambe
```

Massa e tensore d'inerzia del pannello derivano dal suo volume (larghezza × altezza × 1 mm) assumendo una densità da plexiglass/PMMA (`1180 kg/m³`); una mesh a forma di parallelepipedo viene generata automaticamente e salvata in `screen_panel.stl` dentro `mesh_dir`, rigenerata a ogni cambio di dimensione. Il pannello è internamente un unico `opensim.Body` ("screen_panel") saldato al ground con un `WeldJoint` (nessun grado di libertà): la sua posa è interamente determinata da `center_x`/`center_y`/`center_z`/`angle_deg`, oppure da `rotate()`/`translate()` (thin wrapper sulle omonime funzioni di `operators`, applicate al container privato che lo rappresenta).

`mesh_dir` (opzionale, passabile anche al costruttore) è la cartella dove viene scritto `screen_panel.stl`: di default è `components/assets/meshes/` dentro il package stesso (creata automaticamente se mancante), ma può essere impostata su un percorso qualunque -- utile ad es. quando quella cartella di default non è scrivibile (un'installazione del package in una posizione protetta) o per raccogliere altrove le mesh generate da un modello. `set_mesh_dir` sposta la destinazione e rigenera subito la mesh lì; quella scritta in precedenza nella vecchia cartella non viene rimossa:

```python
screen = Screen(mesh_dir="C:/tmp/mie_mesh")
screen.set_mesh_dir("C:/tmp/altra_cartella")  # creata automaticamente se non esiste
```

## Creare un parallelepipedo (Box)

`Box` è un parallelepipedo rigido generico, pensato per rappresentare un oggetto/ingombro qualunque nella scena (es. un elemento di un attrezzo) quando non serve altro che la sua geometria, massa e posa. Come `Screen`, è un **componente** (sottoclasse di `components.Body`): nessun `show()`/`export()` proprio, va aggiunto a un container per essere visualizzato:

```python
from opensim_models import Box

box = Box(
    width=0.2, height=0.4, depth=0.1,       # metri, lungo gli assi locali X/Y/Z
    origin=(0.0, 1.0, 0.0),                 # centro nel ground frame, in metri
    angle_deg=(0.0, 0.0, 30.0),             # Eulero X-Y-Z body-fixed, in gradi, attorno agli assi del ground
    mass_kg=1.5,
)
```

A differenza di `Screen`, la massa è un dato diretto (`mass_kg`, non derivata da una densità di materiale); il tensore d'inerzia resta comunque quello analitico di un parallelepipedo omogeneo pieno con quella massa e quelle dimensioni. Ogni dimensione ha una property in lettura (`width`, `height`, `depth`, `mass_kg`) e un setter dedicato (`set_width`, `set_height`, `set_depth`, `set_mass_kg`) che ricostruisce corpo, mesh e giunto -- la mesh, generata con lo stesso writer STL usato internamente da `operators.add_box_body` (`opensim_models._primitives.write_box_mesh`), viene salvata dentro `mesh_dir` e rigenerata a ogni cambio di dimensione. `width`/`height`/`depth`/`mass_kg` devono essere finiti e strettamente positivi, altrimenti il costruttore (o il setter) solleva `ValueError`.

Il costruttore accetta anche `name` (default `None`, cioè `"box"`): il nome OpenSim del corpo, impostabile anche dopo con `set_name` -- a differenza di una rinomina generica, qui il nome resta "sticky" attraverso qualunque rebuild (un setter di dimensione/massa/posa, o `set_name` stesso) e attraverso `copy()`. Determina anche il nome del file mesh (`f"{name}.stl"`, quindi `box.stl` solo per il nome di default): due `Box` con `mesh_dir` condiviso ma `name` diversi scrivono due file distinti invece di sovrascriversi a vicenda -- importante non appena se ne creano più di uno nella stessa cartella (es. più parti di uno stesso attrezzo).

Come `Screen`, `Box` accetta `mesh_dir` al costruttore (property in lettura `mesh_dir`, setter dedicato `set_mesh_dir`): la cartella dove viene scritta la mesh, di default `components/assets/meshes/` dentro il package (creata automaticamente se mancante), sostituibile con un percorso qualunque -- vedi "Creare uno schermo (Screen)" sopra per i dettagli (stesso comportamento per entrambi i componenti, incluso il parametro `name`).

`origin`/`angle_deg` impostano la posa iniziale (un `WeldJoint` verso ground), e hanno anche loro un setter dedicato (`set_origin`, `set_angle_deg`, ciascuno dei due preserva l'altra metà della posa corrente): ma a differenza delle dimensioni, le property stesse vengono sempre lette direttamente dalla posa corrente del corpo, quindi riflettono comunque l'ultima cosa che lo ha spostato -- gli argomenti del costruttore, `set_origin`/`set_angle_deg`, un setter di dimensione (che preserva la posa corrente durante la ricostruzione), oppure `rotate()`/`translate()` -- tutti modi ugualmente validi per riposizionare un `Box` dopo la costruzione:

```python
box.set_angle_deg((0.0, 0.0, 0.0))           # orientamento assoluto, origine invariata
box.rotate(box.com, (0.0, 0.0, 1.0), 90.0)   # rotazione relativa di 90° attorno al proprio centro di massa
box.translate((0.0, 0.5, 0.0))               # traslazione relativa

print(box.origin)     # riflette già tutti i passaggi sopra
print(box.angle_deg)
```

`com` (centro di massa, calcolato con lo stesso meccanismo nativo OpenSim usato da `User.com`) coincide con `origin` per un corpo singolo e omogeneo, ma è esposto a parte proprio perché è il perno naturale per `rotate()`. `corners` restituisce le coordinate nel ground frame degli 8 spigoli del parallelepipedo (tutte le combinazioni di ±metà misura lungo gli assi locali correnti, trasformate nella posa attuale) -- anch'esso sempre coerente con l'ultima posa, utile per verifiche di ingombro/allineamento con altra geometria della scena:

```python
for corner in box.corners:
    print(corner)
```

Come `Screen`, `Box` è internamente un unico `opensim.Body` ("box") saldato al ground con un `WeldJoint` (nessun grado di libertà): non ha una `postura` articolare propria, è pensato per essere posizionato/orientato rigidamente, non animato internamente.

## Comporre più modelli

Due o più `OpenSimModel` (ad esempio due `User`) possono essere combinati in un unico modello OpenSim esportabile:

```python
combined = user_model_a + user_model_b
combined.export("combined.osim")
```

`combined` è un `OpenSimModel` generico che contiene tutti i componenti di entrambi gli operandi (corpi, giunti, muscoli/forze, marker, vincoli); nessuno dei due operandi originali viene modificato, e il risultato non è mai una sottoclasse di uno dei due (anche `user_a + user_b` è un `OpenSimModel` generico, non uno `User`). Se un componente del secondo modello ha lo stesso nome di uno già presente nel primo, viene rinominato automaticamente con un prefisso (il nome della classe dell'operando, oppure un prefisso esplicito tramite `add_model(..., name=...)`). I giunti agganciati al `ground` nei modelli sorgente restano agganciati al ground condiviso del modello combinato, così i due modelli mantengono la propria collocazione di default.

Un **componente** standalone come `Box`/`Screen` (non un `OpenSimModel`, ma dotato di un container interno privato) si aggiunge con lo stesso `+`, in entrambe le direzioni, con lo stesso risultato -- un `OpenSimModel` fuso, pronto per `show()`/`export()`:

```python
scena = user_model + screen       # oppure screen + user_model: stesso risultato
"screen_panel" in scena.bodies    # True
"screen_panel" in user_model.bodies  # False: user_model non è stato modificato
scena.show()
```

Per comporre più di due modelli/componenti, `+` si può concatenare (`a + b + c`), oppure si può operare in place su un modello esistente:

```python
scene = OpenSimModel(model_path=None)
scene.add_model(user_model)
...
scene.remove_model(user_model)  # torna allo stato precedente
```

`add_model`/`remove_model` richiedono sempre un `OpenSimModel`; `+`/la sua forma riflessa accettano anche un componente standalone come `Box`/`Screen` (vedi sopra). In entrambi i casi, un operando del tipo sbagliato solleva `TypeError`. `remove_model` richiede che il modello indicato sia stato effettivamente aggiunto con `add_model` in precedenza, altrimenti solleva `ValueError`.

### Agganciare un componente a un altro corpo (`attach_component`)

Un `Box`/`Screen` appena fuso in un modello (`model + box`) eredita il proprio giunto originale (saldato al ground del componente stesso, prima della fusione). Per agganciarlo invece a un body *del modello risultante* -- es. il torso di uno `User`, non il ground condiviso -- usa `model.attach_component`:

```python
merged = user_model + box
merged.attach_component(
    merged.body("box"), to=merged.body("torso"),
    child_point="com", parent_point=(0.0, 0.1, 0.0),  # 10 cm sopra il com del torso
    joint_type="weld",
)
```

`attach_component` rimuove il giunto attuale di `child` e ne crea uno nuovo (stesso tipo di `add_free_joint`/`add_pin_joint`/.../`add_weld_joint`, scelto con `joint_type`) verso `to`, nel punto indicato da `child_point`/`parent_point` -- `"com"` (default, centro di massa) oppure una tupla `(x, y, z)` esplicita nel frame locale del corpo. `child`/`to` devono già appartenere allo stesso modello (un `Joint` OpenSim non può mai collegare due `opensim.Model` diversi): per un componente standalone, questo significa fonderlo prima con `+`, come nell'esempio sopra. Solleva `ValueError` se `child` non è collegato da nessun giunto nel modello (es. non è mai stato fuso, o il suo giunto è già stato rimosso).

`child_orientation_deg`/`parent_orientation_deg` (Eulero X-Y-Z body-fixed, in gradi, default nessuna inclinazione) orientano il nuovo giunto sui due lati -- indispensabile per un tipo con un asse non simmetrico come `"slider"` (scorre lungo il proprio asse X locale): per vincolare un corpo a scorrere lungo una retta inclinata di `incline_deg` rispetto al piano del genitore,

```python
full.attach_component(
    full.body("box"), to=full.ground,
    parent_orientation_deg=(0.0, 0.0, incline_deg),
    joint_type="slider",
)
```

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

- `tests/test_model.py` copre `OpenSimModel` in modo esaustivo: caricamento (da file, vuoto, file mancante), sblocco delle coordinate, accessori nominati, gestione di coordinate (posizione, velocità)/marker/muscoli/massa dei corpi, `position_global`/`position_local` su `Body`/`Marker`/`Joint` (verificati contro le chiamate OpenSim native equivalenti), `Joint.parent_frame`/`child_frame` e il wrapper `OffsetFrame` risultante (`translation`/`set_translation`, `orientation_deg`/`set_orientation_deg`, relativa validazione), `parents` in avanti (`Marker`, `Joint`, `OffsetFrame`) e all'indietro (`Body`, verificato sul modello Rajagopal reale: `femur_r` trova esattamente i 3 giunti, i muscoli e i marker attesi, e nessun vincolo/forza generici), `update_state()` (propagazione a quantità derivate, comportamento "grezzo" dei setter), `reinitialize()` (inclusa la coerenza di coordinate accoppiate da un `CoordinateCouplerConstraint`), `copy()` (indipendenza del modello copiato, preservazione di postura e di attributi delle sottoclassi), scaling, export (inclusa la copia delle mesh in `Geometry/`), cartelle di geometria, `rotate()`/`translate()` (corretta delega a `operators.rotate_object`/`translate_object`, inclusa la copia indipendente restituita con `inplace=False`), `add_contact_sphere`/`add_contact_half_space`/`add_offset_frame` (corretta delega a `operators`, dispatch di `model.contact_geometries` verso il wrapper specifico) e l'intera composizione di modelli (`add_model`, `remove_model`, `__add__`, `__radd__`, rinomina automatica sulle collisioni, preservazione della postura degli operandi, controlli di tipo).
- `tests/test_model_wrapper_signatures.py` verifica che ognuno dei 26 convenience method di `OpenSimModel` che delegano a `opensim_models.operators` (`add_body`, `add_weld_joint`, `attach_component`, ...) abbia esattamente la stessa firma (nomi, ordine, default) della funzione omonima in `operators` -- una rete di sicurezza contro il disallineamento fra le due, che altrimenti nessun test rileverebbe.
- `tests/test_user.py` copre `User` in modo esaustivo: caricamento e validazione dei dati ANSUR (eseguibili anche senza OpenSim installato), risoluzione di percentile/altezza (inclusa l'estrapolazione PCHIP fuori range con `UserWarning`), scaling antropometrico, massa totale (default dal `weight_kg` ANSUR risolto, override esplicito con `mass_kg`, invarianza della geometria dei segmenti rispetto alla massa, fattore di `scaleMass` uniforme su ogni corpo, validazione, `copy()`, e la ricalibrazione post-costruzione con `set_mass_kg` inclusa la preservazione della postura), ogni singolo setter di postura e la relativa property di lettura, ogni centro articolare (confrontato con la posizione OpenSim nativa) e il dizionario `joint_centers`, le misure derivate geometricamente (lunghezze di coscia/gamba/braccio/avambraccio, altezza del tronco, larghezza spalle) e quelle lette direttamente da ANSUR (circonferenze, profondità, larghezze, inclusa la stima a sezione circolare di coscia/polpaccio), `com`/`cop`/`set_position`, e l'integrazione con la facade ereditata da `OpenSimModel`.
- `tests/test_user_generated_code_is_fresh.py` verifica che `_posture_generated.py`/`_joint_centers_generated.py` corrispondano esattamente a quanto produrrebbe `scripts/generate_user_code.py` a partire dalle tabelle correnti, e che i due mixin espongano esattamente i nomi attesi dalle tabelle -- fallisce se una tabella viene modificata senza rigenerare i file.
- `tests/test_screen.py` copre `Screen` in modo esaustivo: dimensionamento (esplicito, da diagonale, priorità e fallback tra i due), posa (`center_*`/`angle_deg`, `origin`/`set_origin` -- quest'ultimo analogo a `Box`), `position_global`/`position_local`, struttura del modello (un corpo, un `WeldJoint`), rigenerazione della mesh sui setter e registrazione della cartella di geometria.
- `tests/test_box.py` copre `Box` in modo esaustivo: massa/inerzia (default, esplicita, analitica per un parallelepipedo pieno) e relativa validazione (dimensioni/massa non positive), `origin`/`angle_deg`/`com` alla costruzione (inclusa la loro equivalenza con `position_global`/`inclination` ereditati, di cui sono ora alias), `set_origin`/`set_angle_deg` (ciascuno preserva l'altra metà della posa), il fatto che `origin`/`angle_deg`/`corners` restino sempre coerenti con la posa corrente (anche dopo `rotate()`/`translate()` ereditati, e attraverso un setter di dimensione, che preserva la posa durante la ricostruzione), gli 8 spigoli (valori attesi e comportamento rigido sotto traslazione), struttura del modello (un corpo, un `WeldJoint`), indipendenza delle istanze, `copy()`, e rigenerazione della mesh sui setter.
- `tests/test_cad_import.py` copre `OpenSimModel.from_step`: massa/inerzia calcolate correttamente da un solido di riferimento (con conversione di unità), generazione della mesh, giunti verso ground di default, combinazione di più solidi in un unico corpo (`as_one_object`, di default e disattivata) e relativi errori (file mancante, STEP senza solidi).
- `tests/test_operators.py` copre `opensim_models.operators`: `add_component`/`remove_component` generici e i relativi errori, i wrapper nominati per corpi/giunti/forze-muscoli/marker/vincoli, i costruttori di giunto nominati (gradi di libertà, posizione/orientamento), i corpi a forma primitiva (massa/inerzia analitiche, geometria nativa vs mesh generata, tipo di giunto, batching), il collegamento di mesh esistenti a un corpo, l'uso di `structural_change()` per un batch di modifiche correlate (corpo+giunto), la preservazione della postura delle coordinate non toccate dalla modifica strutturale, `position_global`/`position_local` su `ContactGeometry`, il dispatch di `add_weld_constraint`/`add_point_constraint`/`add_point_on_plane_constraint`/`add_sliding_point_contact` (e di `model.constraints`/`model.forces`) verso il wrapper più specifico (`WeldConstraint`, `PointConstraint`, `ConstantDistanceConstraint`, `ExponentialContactForce`, con i rispettivi punti verificati numericamente, inclusa una regressione sul bug per cui un vincolo/forza riletto da `model.constraints`/`model.forces` -- a differenza di uno appena restituito da `add_*` -- veniva wrappato nel tipo specifico ma con `.raw` ancora genericamente tipato, rompendo silenziosamente `.parents`/i suoi punti dietro un `getattr(..., "parents", ())`), `parents` in avanti su `Muscle` (deduplica i body del percorso) e `ContactGeometry`, e `parents` sui tre sottotipi di `Constraint` verificato anche "all'indietro" tramite `Body.parents`, la conferma che `add_coordinate_coupler_constraint`/un muscolo restino sul wrapper generico (`Constraint`/`Force`, mai promossi), `add_contact_sphere`/`add_contact_half_space`/`add_contact_mesh` (dispatch verso `ContactSphere`/`ContactHalfSpace`/`ContactMesh`, proprietà specifiche verificate numericamente, validazione del raggio, `parents`, e -- test di regressione che crasherebbe l'intero processo di test se il bug tornasse -- la costruzione pezzo-per-pezzo di `ContactMesh` che aggira il crash nativo del suo costruttore "tutto in una volta", sia con un file reale sia con uno inesistente per la risoluzione lazy), `add_offset_frame` (posizione/orientamento/`parents` verificati numericamente, path risultante `/bodyset/<body>/<name>`, utilizzabile a valle come `body=` di `add_marker`/`add_contact_sphere` e come `to=` di `attach_component` col fallback di `parent_point="com"` sull'origine del frame, e sopravvivenza confermata alla fusione `model + other` -- a differenza dell'ancora interna anonima di `add_model`, che resta deliberatamente fuori da ogni `_MERGE_SETS`), e `rotate_object`/`translate_object` (mutazione in place di marker e `PhysicalOffsetFrame`/`OffsetFrame`, sola lettura su `Body`/`Joint`, perno/oggetto esterni come componente o coordinata, funzionamento standalone senza modello, rotazione/traslazione rigida dell'intero modello attraverso i suoi giunti agganciati al ground, `inplace=False` -- copia indipendente del modello o dell'oggetto, originale invariato -- ed i relativi errori).
- `tests/test_operators_public_api.py` verifica che lo split di `operators.py` in package (vedi "Contenuto del progetto") non abbia cambiato la superficie pubblica: `operators.__all__` elenca esattamente gli stessi nomi di prima (`euclidean_distance`, `from_global_to_local`, `from_local_to_global` inclusi), tutti risolvono a un callable, e `from opensim_models.operators import *` funziona ancora.
- `tests/test_solving.py` copre `operators.solve_coordinates`/`operators.solve_point_coincidence` (e i rispettivi metodi delegati `OpenSimModel.solve_coordinates`/`solve_point_coincidence`): il nucleo generico su un modello minimo costruito ad hoc (un corpo con un `PinJoint` e un marker, con cinematica diretta nota in forma chiusa), la coerenza fra il valore riportato (`result.x`) e lo stato effettivamente lasciato sul modello, il default di `x0` al valore corrente della coordinata, la validazione (`coordinate_names`/`point_pairs` vuoti, `x0` di lunghezza sbagliata, forma di `point_pairs` non valida), una non convergenza forzata (`max_nfev=1`) verificata come eccezione ordinaria (o risultato con `success=False`, con `raise_on_failure=False`) e non come crash, entrambe le forme di `point_pairs` (`(getter_a, getter_b)` e `(frame_a, punto_a, frame_b, punto_b)`, quest'ultima nello stesso schema spallaccio/pad o tallone/poggiapiedi di `analisi.py`), e -- sul modello `User` reale, non solo sul modello minimo -- un confronto diretto con un `brentq` costruito a mano (stesse primitive `enforce_constraints=False`/`realizePosition`) sulla stessa condizione, e un recupero multi-coordinata in stile cinematica inversa (postura nota, azzerata, poi ritrovata da una stima vicina).
- `tests/test_geometry.py` copre `operators.euclidean_distance` (un caso noto -- triangolo 3-4-5 --, punti coincidenti, simmetria, input come liste/array numpy, tipo di ritorno, e validazione di dimensione/valori non finiti -- le uniche che non richiedono i binding OpenSim installati, dato che la funzione è pura geometria) e `operators.from_global_to_local`/`operators.from_local_to_global` (un caso noto con un corpo a posa nota -- ruotata e non -- verificato a mano, il round-trip globale->locale->globale e locale->globale->locale, l'accettazione sia del wrapper `Body`/`OffsetFrame` sia del rispettivo oggetto `opensim` grezzo, il caso specifico di `Box` -- il cui container privato viene ricostruito da zero a ogni cambio di posa, a differenza del registro proprietario generico su cui si appoggiano `Body`/`OffsetFrame` -- il rifiuto di `Marker`/`Joint`, che non sono frame orientabili, il rifiuto di un oggetto senza un modello proprietario risolvibile, e la validazione delle coordinate).
- `tests/test_player.py` copre la logica pura (senza una finestra/visualizzatore reale) di `opensim_models._gui.player`: `MotionData` (conversione gradi->radianti solo sulle coordinate rotazionali quando `inDegrees=yes`, nessuna conversione su quelle traslazionali, interpolazione lineare e clamp fuori range, colonne che non sono coordinate del modello, errori su tabelle troppo corte) e `MotionPlayer` (play/pause, stop, cycle, avanti/indietro veloce con i relativi limiti e la ripartenza dal verso opposto, wraparound in avanti/indietro con `loop`, seek, e i relativi errori di costruzione). Il collegamento a una finestra Tk reale, l'incorporamento nativo Win32 della vista 3D, la navigazione della camera, il tooltip ed il salvataggio di immagine/animazione (`opensim_models._gui.export`) non sono automatizzati: richiedono un display e un contesto OpenGL reali, verificati manualmente.

I test che richiedono i binding OpenSim vengono saltati automaticamente se il modulo `opensim` non è importabile; quelli di `test_cad_import.py` vengono saltati se `pythonocc-core` non è importabile; i test sui soli dati ANSUR restano eseguibili in ogni caso.

## Limiti e note

- Lo scaling è completo a livello di pipeline OpenSim, ma la qualità antropometrica dipende dalla corrispondenza tra misura ANSUR e segmento.
- Le misure senza corrispondenza diretta usano il rapporto di statura come fallback esplicito.
- Un'altezza fuori dal range ANSUR osservato viene estrapolata (PCHIP con coda lineare) ed emette un `UserWarning`: più l'altezza richiesta è lontana dal range osservato, meno affidabile è l'estrapolazione.
- Le lunghezze/larghezze/profondità di coscia e polpaccio non hanno un corrispondente ANSUR diretto per tutti gli assi: dove manca una larghezza misurata, la profondità è stimata assumendo una sezione circolare (`diametro = circonferenza / π`), non una misura reale.
- Le lunghezze delle singole dita della mano non sono disponibili: né ANSUR II né il modello OpenSim di `User` le contengono.
- La composizione di modelli (`add_model`/`__add__`) è pensata per scheletri/oggetti indipendenti agganciati al ground: non offre (ancora) un modo per saldare un modello a un body specifico dell'altro.
- Sono richiesti binding OpenSim compatibili con la versione del modello e con l'interprete Python attivo.
- `OpenSimModel.from_step` non deduce alcuna gerarchia cinematica dal file CAD (un file STEP non la contiene): i `FreeJoint` generati di default vanno sostituiti con i giunti reali dell'assieme prima di affidarsi alla dinamica del modello.
- La densità usata da `from_step` è un valore generico in assenza di dati materiale nel file STEP: per una massa/inerzia fisicamente corrette va passata esplicitamente (globalmente o per parte).
- Ogni setter di `Screen` ricostruisce da zero corpo, mesh e giunto: economico per un singolo pannello, ma non pensato per essere chiamato ad alta frequenza (es. in un loop di animazione).
