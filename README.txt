Natural Language Processing -  Word Similarity.

0) running the code: python ex3.py 0/1/2
for sentence/window/dependcy context. default is 1 if none given.
this will print the top 20 for each for needed.
This wiki dataset file should be in the same folder as the code.


1)
 a) Filters and thresholds :
- first features vectors were created for words who show at least 100 times in the corpus(had to pre count)
- to make the feature vectors "shorter" i required context words(features) to also have some minimal
threshold of occurence in the corpus.
for the window context and depedcy context the threshold was 5.
for the full window sentence I used a threshold of 8 because each word had more context words so
it was easier for contex words to cross that threshold.also the time here wa longer than the other 2 contexts
so saving memory was needed here.

- after computing the co-occurence anoter filter was applied where a feature was removed from a feature
vector if it's count was lower than some low number(3-4)

b) 8458 - all context types

c) window context 2467577
   all sentence   7102598
   dependency     13855517



2) full similarity table in the end of the file.
scores(**) , using words piano and hospital:
context     MAP-topic    precision-topic         MAP-semantic                precision-semantic
sentence    7.36         0.75                   1.9114222415441455           0.425
window      6.89         0.75                   3.363                        0.55
dependancy  5.26         0.6                    7.875                        0.755

As we can see some contexts are more oriented towards topic/semantic similarity.
Full sentence context preforms very well on topic similarity , that is because it is the
"broadest" context , for each word the whole sentence is looked at.Sentences usually
talk about one topic , that means words from a single topic will show up with other
words from the same topic alot.this effects first order similarity but also 2ed order here,
for example sentences like : "received medial health care at the hospital"
hospital and care PMI will increase but also the 2ed order similarity as both words
will incrase their PMI with the rest of the words(e.g medical).
Sentence context got all the topic right for piano .for piano most words that i tagged as a now
had a "almost yes" feeling.

dependancy context - this context does the worst on topic relation and the best on semanitc.
This was expected as the way the context vectors are created here are from the way the words
are used semanticly in sentences.

Word window is somewhat in between them.
We can see that the window context is less adventures with the context it outputs and give context that
is closer to the word itself,for example for piano in a 2 window context , most words are other instruments
and those who are not are typical "music words". If we look at full sentences context
it is a bit more wild with words like percussion and ensemble.


(**) I was not sure how you define precision at K.this is how i calculated the AP:
    right = 0 ; score = 0.0
    for i in range(len(arr)):
        if arr[i] == 1:
            right +=1
        prec =float( right) / (i+1)
        recall = float(right) / len(arr)
        score += prec*recall


 manual annotations(words were removed from their own lists):
			--- hospital ---
sentence context			topic-related		semantic class     
medical     				yes                  no 
care                        yes                  no
health                      yes                  no
private                     yes                  no
board                       yes 				 no                 
institute                   no                   yes
mission                     no                   no
staff                       yes                  no
centre                      yes                   yes
found                       no                   yes
locate                      no                   yes
raise                       no                   no               
practice                    yes                  no
education                   no                   yes     
center                      yes					 yes
spend                       no                   no
recently                    no                   no
addition                    no                   no
facility                    no                   yes
prior                       yes                  no


------------------- hospital-------------------------
window=2 context         topic-related		semantic class 
medical                  yes                no
care                     yes                no
health                   yes                no
center                   yes                no
addition                 no                 yes
university               no                 yes
office                   no                 yes
facility                 no                 yes
patient                  yes                no
private                  yes                no
village                  no                 yes
focus                    no                 no      
home                     no                 yes
education                yes                no
college                  no                 yes
science                  yes                no
public                   yes                yes
york                     no                 yes
board                    yes                no
art                      no                 no


------------------- hospital-------------------------
dependency         topic-related		semantic class     
library            no                    yes
facility           no                    yes  
beginning          no                    no
office             no                    yes
foundation         yes                   yes
institution        yes                   yes
lot                yes                   yes                 
municipality       no                    yes
variety            yes                   yes
example            no                    no
museum             no                    yes
branch             yes                   yes 
parish             yes                   yes
court              no                    yes
center             yes                   yes
studio             no                    yes
commander          no                    no
village            no                    yes
wikipedia          no                    no
bank               no                    yes


					--- piano ---				
sentence context   topic-related		semantic class
violin      		  yes				yes
cello      			  yes				yes
solo                  yes				no
bass     			  yes				yes
orchestra             yes               no
flute                 yes				yes
sonata                yes               no
composer              yes               no
instrument            yes               no
guitar                yes               yes
composition           yes               yes
string                yes               no
trio                  yes               no
op   				  yes	            no
drum                  yes				yes
keyboard              yes				yes
quartet               yes               no
concerto              yes               no
percussion            yes               yes
ensemble              yes               yes


------------------- piano-------------------------
window=2 context		topic-related		semantic class
violin                  yes					yes
flute                   yes					yes
cello                   yes					yes
sonata                  yes                 no
saxophone               yes					yes
concerto                yes                 no
op                      yes                 no
viola                   yes					yes    
guitar                  yes					yes
solo                    yes                 no
bass                    yes					yes
keyboard                yes					yes      
trumpet                 yes					yes
string                  yes					yes
soloist                 yes                 no
drum                    yes					yes
orchestra               yes                 no
vocal                   yes                 no
instrument              yes                 yes
organ                   yes					yes


------------------- piano-------------------------
dependency		topic-related		semantic class 
violin           yes                 yes
cello            yes                 yes      
viola            yes                 yes
guitar           yes                 yes
bass             yes                 yes
horn             yes                 yes
saxophone        yes                 yes
flute            yes                 yes
keyboard         yes                 yes
drum             yes                 yes
remainder        no                  yes   
trumpet          yes                 yes
instrument       yes                 yes
vocal            yes                 no
formulation      no                  no   
guitarist        yes                 no
orchestra        yes                 no
finale           yes                 yes
organ            yes                 yes
duchy            no					 no




PMI : functions - calc_PMI
I calculated PMI(x,y) as P(X|Y) / P(X) = (|(X,Y)|/|(Y,*)) / ( (|(Y,*)| / | (* , *)| ))
I had a hash table that counted for each word X how many times it was seen in a context - |(X,*)|
so that is how i obtains |(X,*)| and I obtained |(*,*)| by just summing |(X,*)| for all X.

Matrix Multipication - functions - sparse_matrix_mult.

A vector(really a hashtable) was created for each feature, that is for "rows" in the matrix.(it can be done while creating
the words(who appear 100+ times) vectors , but i choose to do it after the PMI were created to save some time.
I did it by going over the word vectors and creating vectors for its features(context) , making use of PMI symmetry.



scores , using words piano and hospital:
context     MAP-topic    precision-topic         MAP-semantic                precision-semantic
window      9.5          0.952                   7.3                         0.75
dependancy  8.33         0.8                     10.1                        0.975

As we can see again window contexts does better on topic relation(window was 5 here so its actully getting closer to sentence context).
That is for the same reason as explained above , window prefering topic and dependcy semantics.

We can see W2vec getting slightly better results than my preformence.
I see 2 reasons for that:
My code cleans up rare words , both as contexts and word vectors,
w2vec finds some very esoteric words like leprosarium , violoncello which my
code cannot find and should be easy to find as they probably appear only in the context of a specific topic.
Another reason is that w2vec finds words who actully should have the same lemma form like ranking hospitals for hospital.

Mannual annonations:(full list at end of file)

-------hospital------
window=5 context         topic-related		semantic class
clinic					 yes				yes
hospitals				 yes				yes
infirmary				 yes				yes
hospice         		 yes				yes
lying-in                 yes                no
dispensary      		 yes				yes
polyclinic      		 yes				yes
sanatorium      		 yes				yes
convalescent             yes                no
mulago                   no                 yes
addenbrooke              yes				yes 
bethlem                  no                 no
psychiatric              yes                no
maudsley                 yes                yes
siriraj                  yes                yes
sanitarium               yes				yes
in-patient               yes                no
incurables               yes                no
orthopaedic              yes                no
westmead                 yes                yes


-------hospital------
dependency context         topic-related		semantic class
sanatorium                  yes                  yes
hospice                     yes                  yes
sanitorium                  yes                  yes
hospitals                   yes                  yes
sanitarium                  yes                  yes
clinic                      yes                  yes
infirmary                   yes                  yes
polyclinic                   yes                 yes
dispensary                   yes                 yes
orphanage                    no                  yes
poorhouse                    no                  yes
almshouse                    no                  yes
workhouse                    no                  yes 
institutet                   no                  no
leprosarium                  yes                 yes
rikshospitalet               yes                 yes
heliport                     no                  yes                    
gaol                         yes                 yes
guesthouse                   no                  yes
motherhouse                  no                  yes



-------piano------
window=5 context         topic-related		semantic class
violin                   yes                yes
cello                    yes                yes
harpsichord              yes                yes
clarinet                 yes                yes
viola                    yes                yes
flute                    yes                yes
bassoon                  yes                yes
violoncello              yes                yes
oboe                     yes                yes
concerto                 yes                no
saxophone                yes                yes
accordion                yes                yes
harp                     yes                yes
trombone                 yes                yes
sonatas                  yes                no
trumpet                  yes                yes
mandolin                 yes                yes
pianoforte               yes                yes
vibraphone               yes                yes
concertos                no                 no


-------piano------
dependency context         topic-related		semantic class
violin                     yes                  yes
cello                      yes                  yes
harpsichord                yes                  yes
saxophone                  yes                  yes
clarinet                   yes                  yes
guitar                     yes                  yes
trombone                    yes                  yes
mandolin                   yes                  yes
vibraphone                 yes                  yes
marimba                    yes                  yes
accordion                  yes                  yes
pianoforte                 yes                  yes
bassoon                    yes                  yes
fortepiano                  yes                  yes
violoncello                 yes                  yes
trumpet                    yes                  yes
harmonica                  yes                  yes
clavinet                   yes                  yes
clavichord                 yes                  yes
euphonium                  yes                  yes







					APPENDIX - LIST OF WORDS
					
					PART 1 WORD LISTS:
					
					
						------------------- car-------------------------
sentence context                                       window=2 context    dependency
drive                                                  vehicle             vehicle      
race                                                   race                ship      
model                                                  drive               type      
front                                                  train               side      
vehicle                                                addition            website      
engine                                                 home                example      
replace                                                ship                site      
driver                                                 available           series      
train                                                  example             production      
six                                                    carry               section      
able                                                   close               collection      
full                                                   replace             level      
carry                                                  design              track      
addition                                               turn                lot      
off                                                    main                others      
next                                                   down                piece      
top                                                    build               train      
track                                                  those               variety      
stop                                                   instead             officer      
ten                                                    hand                class      
						------------------- bus-------------------------
sentence context                                       window=2 context    dependency
route                                                  rail                rail      
operate                                                passenger           stuff      
connect                                                taxi                charter      
train                                                  train               thanks      
transport                                              traffic             edge      
passenger                                              operate             delegate      
rail                                                   commuter            filter      
traffic                                                tram                stub      
railway                                                freight             rest      
express                                                route               pair      
station                                                transit             outline      
via                                                    car                 promise      
vehicle                                                transport           phrase      
transit                                                connect             row      
extend                                                 service             passenger      
platform                                               express             participant      
airport                                                metro               couple      
travel                                                 private             variety      
road                                                   network             crew      
transportation                                         railway             rebel      
						------------------- hospital-------------------------
sentence context                                       window=2 context    dependency
medical                                                medical             library      
care                                                   care                facility      
health                                                 health              beginning      
private                                                center              office      
board                                                  addition            foundation      
institute                                              university          institution      
mission                                                office              lot      
staff                                                  facility            municipality      
centre                                                 patient             variety      
found                                                  private             example      
locate                                                 village             museum      
raise                                                  focus               branch      
practice                                               home                parish      
education                                              education           court      
center                                                 college             center      
spend                                                  science             studio      
recently                                               public              commander      
addition                                               york                village      
facility                                               board               wikipedia      
prior                                                  art                 bank      
						------------------- hotel-------------------------
sentence context                                       window=2 context    dependency
room                                                   resort              kind      
restaurant                                             restaurant          resort      
purchase                                               room                proposal      
hall                                                   casino              theatre      
centre                                                 estate              residence      
facility                                               shop                museum      
historic                                               apartment           gallery      
building                                               office              estate      
stay                                                   hall                others      
construct                                              outside             founder      
owner                                                  complex             uk      
numerous                                               theatre             commune      
beginning                                              building            beginning      
location                                               residence           census      
nearby                                                 private             lot      
store                                                  facility            guideline      
locate                                                 addition            studio      
resident                                               park                rating      
stand                                                  washington          mill      
street                                                 locate              collection      
						------------------- gun-------------------------
sentence context                                       window=2 context    dependency
fire                                                   rifle               artillery      
weapon                                                 artillery           weapon      
machine                                                fire                crew      
heavy                                                  weapon              pair      
arm                                                    cannon              branch      
shoot                                                  machine             stub      
enemy                                                  heavy               soldier      
carry                                                  tank                example      
rifle                                                  battery             battery      
ship                                                   each                lot      
target                                                 enemy               wikipedia      
forward                                                shoot               rest      
command                                                carry               kind      
soldier                                                light               majority      
capture                                                destroy             rifle      
navy                                                   ship                flag      
intend                                                 sniper              edge      
quickly                                                arm                 thanks      
fit                                                    addition            letter      
destroy                                                supply              type      
						------------------- bomb-------------------------
sentence context                                       window=2 context    dependency
destroy                                                bombing             outline      
target                                                 explosive           quote      
shoot                                                  explode             promise      
injure                                                 explosion           wonder      
weapon                                                 enemy               favor      
bomber                                                 ordnance            handle      
fly                                                    torpedo             attribute      
bombing                                                bomber              seal      
civilian                                               destroy             span      
damage                                                 tank                mix      
enemy                                                  sink                demand      
kill                                                   nuclear             slip      
aircraft                                               damage              paint      
crew                                                   fire                burn      
1942                                                   attack              accord      
fighter                                                strike              welcome      
strike                                                 allied              archive      
heavy                                                  carry               doubt      
defence                                                shell               sort      
1945                                                   kill                surprise      
						------------------- horse-------------------------
sentence context                                       window=2 context    dependency
race                                                   breed               variety      
earlier                                                race                proposal      
britain                                                farm                kind      
raise                                                  dog                 beginning      
farm                                                   car                 lot      
owner                                                  sheep               fact      
decade                                                 raise               dog      
rest                                                   thoroughbred        example      
latter                                                 except              half      
beginning                                              guard               enemy      
following                                              animal              wikipedia      
prior                                                  addition            choice      
quickly                                                pull                commander      
arrive                                                 pig                 collaboration      
finally                                                away                parent      
animal                                                 cattle              resident      
soon                                                   canadian            latter      
hundred                                                drive               others      
yet                                                    2nd                 generation      
expect                                                 these               biography      
						------------------- fox-------------------------
sentence context                                       window=2 context    dependency
broadcast                                              cbs                 cbs      
cbs                                                    nbc                 nbc      
channel                                                abc                 cbc      
affiliate                                              pb                  abc      
abc                                                    cnn                 thanks      
network                                                anchor              husband      
morning                                                bbc                 britain      
tv                                                     broadcast           stub      
television                                             television          w      
programming                                            tv                  baker      
news                                                   drama               predecessor      
nbc                                                    news                bennett      
bbc                                                    network             grandson      
coverage                                               channel             cub      
acquire                                                beginning           wilson      
sport                                                  espn                stuff      
schedule                                               interview           retreat      
guest                                                  wolf                scout      
20th                                                   sport               uk      
beginning                                              morning             television      
						------------------- table-------------------------
sentence context                                       window=2 context    dependency
yet                                                    below               variety      
below                                                  tennis              proposal      
following                                              reference           piece      
detail                                                 section             rest      
these                                                  focus               proof      
always                                                 particular          example      
entry                                                  instead             fact      
either                                                 entire              topic      
reference                                              merge               wikipedia      
expect                                                 addition            census      
least                                                  each                beginning      
each                                                   facility            entry      
possible                                               court               edge      
search                                                 tournament          addition      
clear                                                  following           spirit      
content                                                variety             pair      
cite                                                   suggest             stage      
whole                                                  these               wiki      
enough                                                 link                future      
particular                                             either              text      
						------------------- bowl-------------------------
sentence context                                       window=2 context    dependency
super                                                  bowler              super      
nfl                                                    ncaa                guess      
conference                                             super               scout      
tournament                                             batsman             beginning      
ncaa                                                   pace                rest      
coach                                                  all-star            census      
victory                                                tournament          sort      
ball                                                   playoff             log      
earn                                                   slow                hope      
championship                                           baseball            peak      
football                                               ball                thanks      
consecutive                                            cup                 slip      
select                                                 1994                couple      
basketball                                             winner              promise      
defensive                                              pro                 doubt      
baseball                                               1984                pioneer      
pick                                                   victory             wish      
regular                                                quiz                outline      
yard                                                   28                  desire      
florida                                                earn                honour      
						------------------- guitar-------------------------
sentence context                                       window=2 context    dependency
vocal                                                  bass                drum      
bass                                                   drum                bass      
drum                                                   keyboard            keyboard      
solo                                                   piano               vocal      
guitarist                                              vocal               piano      
instrument                                             violin              instrument      
musician                                               acoustic            violin      
piano                                                  instrument          cello      
keyboard                                               saxophone           horn      
string                                                 flute               soundtrack      
sound                                                  guitarist           guitarist      
recording                                              solo                proposal      
acoustic                                               percussion          half      
musical                                                string              saxophone      
percussion                                             rhythm              beginning      
singer                                                 cello               trio      
double                                                 vocalist            lot      
sing                                                   trumpet             website      
band                                                   accompany           afd      
electric                                               jazz                music      
						------------------- piano-------------------------
sentence context                                       window=2 context    dependency
violin                                                 violin              violin      
cello                                                  flute               cello      
solo                                                   cello               viola      
bass                                                   sonata              guitar      
orchestra                                              saxophone           bass      
flute                                                  concerto            horn      
sonata                                                 op                  saxophone      
composer                                               viola               flute      
instrument                                             guitar              keyboard      
guitar                                                 solo                drum      
composition                                            bass                remainder      
string                                                 keyboard            trumpet      
trio                                                   trumpet             instrument      
op                                                     string              vocal      
drum                                                   soloist             formulation      
keyboard                                               drum                guitarist      
quartet                                                orchestra           orchestra      
concerto                                               vocal               finale      
percussion                                             instrument          organ      
ensemble                                               organ               duchy      







*********************  PART 2,WORD2VEC WORD LISTS ************************

Bag of words 5			Dependency
-------car------       -------car------
cars                   truck
truck                  suv
automobile             vehicle
vehicle                minivan
motorbike              cars
motorcycle             speedboat
driver                 racecar
minivan                automobile
suv                    motorcar
lorry                  jeep
motorcar               limousine
mid-engined            minibus
limousine              lorry
front-engined          limo
moped                  motorcycle
motorhome              bike
mercedes-benz          motorhome
bike                   taxicab
rear-engined           roadster
three-wheeled          wagon
-------bus------       -------bus------
buses                  minibus
tram                   tram
metrobus               buses
intercity              jeepney
busses                 taxicab
fixed-route            motorcoach
minibus                taxi
inter-city             trolleybus
ksrtc                  lorry
commuter               truck
apsrtc                 metrobus
msrtc                  streetcar
inter-urban            busses
dial-a-ride            ferryboat
mini-bus               trolley
light-rail             tramcar
rail                   railcar
transit                railmotor
trolleybus             intercityexpress
limited-stop           train
-------hospital------  -------hospital------
clinic                 sanatorium
hospitals              hospice
infirmary              sanitorium
hospice                hospitals
lying-in               sanitarium
dispensary             clinic
polyclinic             infirmary
sanatorium             polyclinic
convalescent           dispensary
mulago                 orphanage
addenbrooke            poorhouse
bethlem                almshouse
psychiatric            workhouse
maudsley               institutet
siriraj                leprosarium
sanitarium             rikshospitalet
in-patient             heliport
incurables             gaol
orthopaedic            guesthouse
westmead               motherhouse
-------hotel------     -------hotel------
motel                  motel
restaurant             hotels
doubletree             casino
sheraton               restaurant
hotels                 inn
ritz-carlton           guesthouse
sofitel                tavern
westin                 cafe
ramada                 ritz-carlton
casino                 nightclub
kempinski              travelodge
mansion                pizzeria
inn                    roadhouse
cafe                   boardinghouse
tavern                 café
apartments             condo
boutique               brewpub
nightclub              sheraton
marriott               steakhouse
travelodge             brasserie
-------gun------       -------gun------
guns                   guns
cannon                 handgun
howitzer               machinegun
sub-machine            howitzer
flamethrower           pistol
belt-fed               rifle
37mm                   shotgun
smoothbore             firearm
pistol                 cannon
shkas                  musket
105mm                  crossbow
40mm                   autocannon
gatling                phaser
recoilless             flamethrower
76mm                   revolver
3-inch                 carbine
rifle                  machine-gun
88mm                   weapon
large-caliber          carronade
autocannons            pounder
-------bomb------      -------bomb------
bombs                  bombs
detonated              firebomb
detonates              landmine
detonate               car-bomb
booby-trap             grenade
detonating             torpedo
firebomb               ied
car-bomb               warhead
exploded               bomber
detonation             bomblets
warhead                missile
500-pound              nuke
b61                    detonator
laser-guided           booby-trap
blast                  kamikaze
explosives             munition
detonations            explosives
landmine               machinegun
tallboy                a-bomb
thermonuclear          firebombs
-------horse------     -------horse------
horses                 horses
standardbred           goat
saddlebred             dog
gelding                stallion
thoroughbred           mule
stallion               bronc
racehorses             cow
dog                    unicycle
riderless              greyhound
gaited                 bareback
bronc                  camel
percheron              appaloosa
pony                   saddlebred
trotting               colt
harness                zebu
chariot                donkey
appaloosa              sidesaddle
sulky                  racehorse
racehorse              elephant
greyhound              pony
-------fox------       -------fox------
abc                    daystar
cbs                    nbc
nbc                    byutv
wxyz-tv                kron
msnbc                  wolf
ctv                    cbs-tv
wsvn                   familynet
familynet              abc
wttg                   wccb
wjbk                   oln
wfxt                   wjar
espn                   hdnet
wofl                   telefutura
cnn                    woodchuck
oln                    nbc-tv
blitzer                soapnet
nesn                   cinemax
wesh                   wdiv
wb                     mundofox
wgn-tv                 coyote
-------table------     -------table------
tables                 tables
sortable               leaderboard
wikitable              sideboard
look-up                chessboard
foosball               textbox
toc                    taskbar
bulleted               gameboard
ping-pong              worksheet
billiard               tray
table-tennis           viewport
textbox                dais
tray                   flowchart
lookup                 playfield
wikitables             mantelpiece
brackets               stepladder
header                 cladogram
footer                 letterbox
tabular                windowsill
menu                   bookcase
carom                  wikitable
-------bowl------      -------bowl------
xliii                  bowls
xlii                   superbowl
xliv                   arenabowl
bowls                  wcws
xlvi                   wnit
tostitos               nlcs
xli                    arenacup
xxxviii                postseason
xlv                    nit
xxxv                   xliii
xxxix                  llws
xxxvii                 beanpot
xlvii                  xlv
xxxvi                  triplemanía
xxxiv                  alcs
bluebonnet             nlds
gator                  cup
xxviii                 kvalserien
xxxii                  tourney
xxxi                   cws
-------guitar------    -------guitar------
harmonica              saxophone
mandolin               bass
bass                   mandolin
drums                  harmonica
guitars                accordion
keyboards              trombone
accordion              violin
banjo                  banjo
saxophone              guitars
12-string              cello
ukulele                piano
trombone               vibraphone
fiddle                 sax
autoharp               trumpet
melodica               autoharp
percussion             clarinet
vibraphone             sitar
tambourine             fiddle
vocals                 drums
fretless               marimba
-------piano------     -------piano------
violin                 violin
cello                  cello
harpsichord            harpsichord
clarinet               saxophone
viola                  clarinet
flute                  guitar
bassoon                trombone
violoncello            mandolin
oboe                   vibraphone
concerto               marimba
saxophone              accordion
accordion              pianoforte
harp                   bassoon
trombone               fortepiano
sonatas                violoncello
trumpet                trumpet
mandolin               harmonica
pianoforte             clavinet
vibraphone             clavichord
concertos              euphonium
