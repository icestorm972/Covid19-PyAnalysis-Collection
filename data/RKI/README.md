# RKI Covid-19 CSV und XLSX
Hier müssen sich die verschiedenen Dateien von der ["Neuartiges Coronavirus" Projektseite auf der RKI Homepage](https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Projekte_RKI/) bzw. deren [GitHub Projektseite](https://github.com/robert-koch-institut/) befinden.

Nur die Github-Daten stehen bisher als Archiv zur Verfügung (wenn auch nur unvollständing). Todesfälle und Altersverteilung stehen (bisher) nichts als Archiv zur Verfügung und wöchentlich beim RKI aktualisiert.
Diese müssen daher selbst von Hand heruntergeladen werden. Aktualisierungstag-Hinweise sind nach Wissensstand 17.11.2021.

## Todesfälle
* **Quelle:** [Todesfälle nach Sterbedatum](https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Projekte_RKI/COVID-19_Todesfaelle.html)
* **Namensschema:** COVID-19_Todesfaelle_RKI-YYYY-MM-DD.xlsx
* **Aktualisierung:** Donnerstags
* **Hinweis:** Benötigt für Zusatzgraph in Verstorbenen-Nowcast

## Altersverteilung
* **Quelle:** [COVID-19-Fälle nach Altersgruppe und Meldewoche](https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Daten/Altersverteilung.html)
* **Namensschema:** Altersverteilung_RKI-YYYY-MM-DD.xlsx
* **Aktualisierung:** Donnerstags
* **Hinweis:** Benötigt für Nicht-Nowcast basierte CFR (z.Z. nicht im GitHub Projekt enthalten)

## Nowcasting
* **Quelle:** [SARS-CoV-2-Nowcasting und -R-Schätzung](https://github.com/robert-koch-institut/SARS-CoV-2-Nowcasting_und_-R-Schaetzung/tree/main/Archiv)
* **Namensschema:** Nowcasting_Zahlen_csv_RKI-YYYY-MM-DD.csv
* **Aktualisierung:** Täglich
* **Hinweis:** Benutzt von *nownowcast.py*

## Hospitalisierungen
* **Quelle:** [COVID-19-Hospitalisierungen in Deutschland](https://github.com/robert-koch-institut/COVID-19-Hospitalisierungen_in_Deutschland/tree/master/Archiv)
* **Namensschema:** YYYY-MM-DD_Deutschland_COVID-19-Hospitalisierungen.csv
* **Aktualisierung:** Täglich
* **Hinweis:** Benutzt von *hospitalization_nowcast2.py*
