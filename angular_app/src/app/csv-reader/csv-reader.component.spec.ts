import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CsvReaderComponent } from './csv-reader.component';

describe('CsvReaderComponent', () => {
  let component: CsvReaderComponent;
  let fixture: ComponentFixture<CsvReaderComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [CsvReaderComponent]
    });
    fixture = TestBed.createComponent(CsvReaderComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
