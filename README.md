# RetailRevenuePrediction

## Projektübersicht
Im Semesterprojekt des Moduls "Data Mining & Grundlagen Maschinelles Lernen 1" entwickeln wir ein Modell zur Umsatzvorhersage für eine deutsche Supermarktkette, um Filialleitungen durch präzise Prognosen bei der Personalplanung zu unterstützen und die Kundenzufriedenheit sowie Kosteneffizienz zu verbessern.

### Die Verzeichnisstruktur sieht wie folgt aus: 
```
RetailRevenuePrediction/
├── data
│   ├── raw                 <- Der ursprüngliche, unveränderliche Datendump.
│   ├── interim             <- Vorläufig verarbeitete Daten.
│   └── processed           <- Endgültig verarbeitete und für die Analyse bereitgestellte Daten
│
├── models
├── notebooks               <- Jupyter Notebooks für Datenanalyse und Modellierung
├── references              <- Alle erläuternden Materialien
├── reports                 <- Erzeugte Analysen als PDF
│   └── figures             <- Erzeugte Grafiken und Abbildungen für die Berichterstattung
│
├── src
│   ├── data                <- Skripte zum Herunterladen oder Generieren von Daten
│   ├── features
│   ├── models
│   └── visualization
│
├── LICENSE
├── README
└── requirements.txt

```

## Nutzung
Um dieses Projekt auszuführen:
```bash
pip install -r requirements.txt
```

Weitere Details zum [Notebook des Machine Learning Model](notebooks/RentalPrediction.ipynb) finden Sie in den jeweiligen Dateien und in der Dokumentation & [Aufgabenstellung](references/DMML1_Projekt_WS23.pdf) .