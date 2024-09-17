import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import * as Papa from 'papaparse';

@Component({
  selector: 'app-csv-reader',
  templateUrl: './csv-reader.component.html',
  styleUrls: ['./csv-reader.component.css']
})
export class CsvReaderComponent implements OnInit {
  public csvData: { header: string, summary: string, category: string }[] = [];

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    this.loadCSV();
  }

  loadCSV() {
    const csvUrl = 'https://complu-bucket.s3.eu-north-1.amazonaws.com/summaries/news_summaries.csv';
    
    // Agregar un parámetro de consulta único para evitar el caché
    const timestamp = new Date().getTime(); // Marca de tiempo en milisegundos
    const urlWithCacheBuster = `${csvUrl}?_=${timestamp}`;

    this.http.get(urlWithCacheBuster, { responseType: 'text' }).subscribe(
      data => {
        console.log('Raw CSV Data:', data);
        Papa.parse(data, {
          header: true,  
          skipEmptyLines: true,
          complete: (result) => {

         console.log('Raw CSV Data:', result);
            this.csvData = result.data
              .filter((row: any) => row.summary && row.summary.trim() !== '')
              .map((row: any) => ({
                header: row.title,
                summary: row.summary,
                category: row.category
              }));
          }
        });
      },
      error => {
        console.error('Error loading the CSV file:', error);
      }
    );
  }

  getBackgroundClass(category: string): string {
    // Cambia las clases de fondo según la categoría
    switch (category) {
      case 'Technology': return 'tm-bgcolor-technology';
      case 'Sports': return 'tm-bgcolor-sports';
      case 'Economy': return 'tm-bgcolor-economy';
      case 'Education': return 'tm-bgcolor-education';  // Color relacionado con educación
      case 'Conflict': return 'tm-bgcolor-conflict';    // Color relacionado con conflictos
      case 'Politics': return 'tm-bgcolor-politics';    // Color relacionado con política
      case 'Justice': return 'tm-bgcolor-justice';      // Color relacionado con justicia
      case 'Incidents': return 'tm-bgcolor-incidents';  // Color relacionado con incidentes
      case 'Global': return 'tm-bgcolor-global';        // Color relacionado con global
      default: return 'tm-bgcolor-3'; // Color por defecto
    }
  }
  
  getIconClass(category: string): string {
    // Cambia el icono según la categoría
    switch (category) {
      case 'Technology': return 'fa fa-laptop';
      case 'Sports': return 'fa fa-futbol-o';
      case 'Economy': return 'fa fa-balance-scale';
      case 'Education': return 'fa fa-graduation-cap';   // Icono relacionado con educación
      case 'Conflict': return 'fa fa-exclamation-triangle'; // Icono de conflicto
      case 'Politics': return 'fa fa-university';        // Icono relacionado con política
      case 'Justice': return 'fa fa-gavel';              // Icono de justicia
      case 'Incidents': return 'fa fa-ambulance';        // Icono relacionado con incidentes
      case 'Global': return 'fa fa-globe';               // Icono relacionado con global
      default: return 'fa fa-newspaper-o'; // Icono por defecto
    }
  }

  getNameClass(category: string): string {
    switch (category) {
      case 'Technology': return 'Technology';
      case 'Sports': return 'Spors';
      case 'Economy': return 'Economy';
      case 'Education': return 'Education';
      case 'Conflict': return 'Conflict';
      case 'Politics': return 'Politics';
      case 'Justice': return 'Justice';
      case 'Incidents': return 'Incidents';
      case 'Global': return 'Global';
      default: return 'News'; // Nombre por defecto
    }
  }  
}
