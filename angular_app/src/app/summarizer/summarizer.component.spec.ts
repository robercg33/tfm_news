import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SummarizerComponent } from './summarizer.component';

describe('SummarizerComponent', () => {
  let component: SummarizerComponent;
  let fixture: ComponentFixture<SummarizerComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [SummarizerComponent]
    });
    fixture = TestBed.createComponent(SummarizerComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
