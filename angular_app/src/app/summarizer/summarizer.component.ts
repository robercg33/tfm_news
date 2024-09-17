import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-summarizer',
  templateUrl: './summarizer.component.html',
  styleUrls: ['./summarizer.component.css']
})
export class SummarizerComponent {
  inputText: string = '';
  summary: string | null = null;
  private apiUrl = 'http://13.48.138.131/summarize';

  constructor(private http: HttpClient) {}

  summarize() {
    const payload = { text: this.inputText };
    console.log('Prediction Result:', payload);
    this.predict(payload.text);

    const body = { text: 'Angular POST Request Example' };
    const headers = { 'Authorization': 'Bearer my-token', 'My-Custom-Header': 'foobar', 'Content-Type': 'application/json' };
    this.http.post<any>(this.apiUrl, body, { headers }).subscribe(data => {
        console.log("kris the output", data);
    });
  }
  predict(text: string): Observable<any> {
    return this.http.post<any>(this.apiUrl, { text });
  }

}