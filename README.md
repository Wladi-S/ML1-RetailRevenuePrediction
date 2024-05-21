# RetailRevenuePrediction

## Projektübersicht
Im Rahmen des Semesterprojekts für das Modul "Data Mining & Grundlagen Maschinelles Lernen 1" widmen wir uns einer praxisnahen Aufgabe – der Umsatzvorhersage für eine große Supermarktkette in Deutschland. Unser Ziel ist es, ein zuverlässiges Vorhersagemodell zu entwickeln, das auf Basis historischer Daten und verschiedener Einflussfaktoren den erwarteten Tagesumsatz jeder Filiale prognostiziert. Eine genaue Vorhersage ermöglicht es den Filialleitungen, die Anzahl der Mitarbeiterinnen und Mitarbeiter optimal zu planen und so die Kundenzufriedenheit zu maximieren, während gleichzeitig die Personalkosten minimiert werden.


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