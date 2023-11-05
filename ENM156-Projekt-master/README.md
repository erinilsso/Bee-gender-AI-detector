# ENM156-Projekt

Flytta de fyra mapparna med bilder (`pollen_detection_data/images`, `Raw data/Pictures Ericsson/Pictures`, `Raw data/Pictures LE/pictures_hive1` och `Raw data/Pictures LE/pictures_hive2`) in i `data`-mappen (om de inte redan finns med). 
I mappen med bilder som du ansvarar för ska en mapp vid namn `labels` ligga innehållande de `.xml`-filer som lagrar labels för dina bilder. 
Se till att inte skriva över något som redan finns med! 
När du pushar dina ändringar kommer `labels`-mappen att följa med, men inte bilderna (det finns ingen anledning att synca bilderna när alla redan har dem lokalt).

I slutändan kommer alla ha all data och alla labels tillgängliga. 

OBS: Det är viktigt att hela filsökvägen bevaras. Dvs., om du jobbar i `Pictures Ericsson/Pictures` så ska även mappen `Pictures Ericsson` ligga i `data`-mappen.

Filerna här ska se ut så här:
```
├── data
|  ├── pollen_detection_data
|  |  └── images
|  |     └── labels
|  └── Raw data
|     ├── Pictures Ericsson
|     |  └── Pictures
|     |     └── labels
|     └── Pictures LE
|        ├── pictures_hive1
|        |  └── labels
|        └── pictures_hive2
|           └── labels
├── src
|  └── ...
├── .gitignore
├── README.md
└── data.csv
```

Filerna på *din* dator ska se ut så här:
```
├── data
|  ├── pollen_detection_data
|  |  └── images
|  |     ├── <bilder>
|  |     └── labels
|  └── Raw data
|     ├── Pictures Ericsson
|     |  └── Pictures
|     |     ├── <bilder>
|     |     └── labels
|     └── Pictures LE
|        ├── pictures_hive1
|        |  ├── <bilder>
|        |  └── labels
|        └── pictures_hive2
|           ├── <bilder>
|           └── labels
├── src
|  └── ...
├── .gitignore
├── README.md
└── data.csv
```
