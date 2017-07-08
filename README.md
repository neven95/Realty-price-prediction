# Realty-price-prediction

#### Ko su članovi tima?

    Neven Jović ra18/2014, grupa3
    Nikola Skrobonja ra25/2014, grupa3

#### Problem koji se rešava:

Problem koji se rešava je predikcija cena nekretnina na ruskom tržistu. Jedni od najvećih troškova kod većine običnih kupaca, ali i investitora, predstavlja kupovina/investiranje u nekretnine, a imajući u vidu nestabilnu rusku ekonomiju, nepredvidive fluktacije cena mogu napraviti veliki problem, kako običnim kupcima, tako i investitorima.
Ova predikcija bi pomogla ljudima koji ulažu, kupuju ili iznajmljuju nekretninu, da imaju više sigurnosti u svoj uloženi novac.
Kompleksne veze između karakteristika nekretnine, kao što je broj soba i lokacija, su dovoljne da učine predikciju komplikovanom. Još kada se svemu tome doda nestabilna ekonomija, potrebno je više od obične regresije da se reši problem i pomogne investitorima i običnim kupcima.


#### Algoritmi koji su se koristili:

Koriscena je tehnika "Stacking".Za tu tehniku meta modele cine XGBOOST, RandomForest i ExtraTrees .Napravljena je Neuronska mreza koja vrsi predikciju na osnovu predikcija meta modela. Pravljenje modela nalazi se u odvojenim fajlovima, dok se u fajl Predikcija ucitavaju gotovi modeli i koriste se dalje za "Stacking" .

#### Podaci koji se koriste:

Pošto je ovaj problem postavljen od strane Sberbank-a, na sajtu Kaggle, u vidu takmičenja, obezbeđeni su veliki setovi podataka. Podaci zahtevaju temeljno i nimalo naivno "uređivanje" .

    train.csv, test.csv: informacije o individualnim transakcijama. Redovi su indeksirani po "id" polju, koji se odnosi na individualnu transakciju.

    macro.csv: podaci o ruskoj makroekonomiji i finansijskom sektoru

    sample_submission.csv: primer ispravnog formata fajla koji se podnosi

    data_dictionary.txt: objašnjenje polja koja su data u ostalim fajlovima

Podaci se u notebook-u "MyExplore.ipynb" ciste i generisu se novi fajlovi.

#### Validacija:

Validacija se vrsi kroz kros validaciju koja se racuna prilikom "fitovanja" modela na podatke. Modeli se dobija na osnovu model_selection-a, gde se preko RandomizedSearchCV nalaze vise modela, koji se posle "fituju" na podatke i za svaki se radi krosvalidacija koja daje skor svakog modela, posle odradjenog "fitovanja" uzimamo model sa najboljim "score-om".

