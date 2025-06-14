/*Programare SAS: combinate sau individuale, minim 8 dintre următoarele facilităţi: crearea unui set de date SAS
din fișiere externe, crearea și folosirea de formate definite de utilizator,
procesarea iterativă și condițională a datelor, crearea de subseturi de date, utilizarea de funcții SAS,
combinarea seturilor de date prin proceduri specifice SAS și SQL, utilizarea de masive,
utilizarea de proceduri pentru raportare, folosirea de proceduri statistice, generarea de grafice,
SAS ML, SAS Viya si altele.*/

/*crearea unui set de date SAS din fișiere externe*/
data date;
	infile'/home/u64242353/data_sas.csv' dsd;
	length country $15; /*setam length 15, pt a incapea cel mai lung nume, slovak republic - 15 caractere*/
	input country $ life_expectancy_at_birth mortality_rate_infant current_health_expenditure
	total_air_emissions unmet_medical_needs total_unemployment_rate hipc self_perceived_health at_risk_of_poverty_rate;
run;

/*crearea si folosirea de formate definite de utilizator */
proc format;
*format denumire tari in functie de regiune;
    value $region_fmt
        'Belgium', 'France', 'Germany', 'Luxembourg', 'Netherlands' = 'Europa de Vest'
        'Bulgaria', 'Romania', 'Hungary', 'Poland', 'Slovak Republic', 'Czechia' = 'Europa de Est'
        'Sweden', 'Denmark', 'Norway', 'Finland', 'Iceland' = 'Europa de Nord'
        'Italy', 'Spain', 'Portugal', 'Greece', 'Malta', 'Cyprus' = 'Europa de Sud'
        'Austria', 'Slovenia', 'Switzerland', 'Serbia' = 'Europa Centrala'
        'Latvia', 'Lithuania', 'Estonia'='Europa Baltica'
        'United Kingdom', 'Ireland', 'Scotland' = 'Insule Britanice'
        other = 'Altele';
*format speranta de viata;
    value life_fmt
        low -< 75 = 'Scăzută (<75 ani)'
    	75 -< 80 = 'Medie (75–80 ani)'
    	80 - high = 'Ridicată (≥80 ani)';
run;

*calculare medie hipc;
proc means data=date noprint;
    var hipc;
    output out=medie_hipc mean=media_hipc;
run;

data _null_;
    set medie_hipc;
    call symputx('media_hipc', media_hipc); /* creeaza macrovariabila media_hipc */
run;

*format in functie de media hipc;
proc format;
    value hipc_fmt
        low - <&media_hipc = 'Sub medie'
        &media_hipc = 'Egal cu media'
        &media_hipc - high = 'Peste medie';
run;


*printare date cu format regiuni;
title "Printare cu format regiuni";
proc print data=date;
format country $region_fmt.;
run;

*printare date cu format life_expectancy;
title "Printare cu format life_expectancy";
proc print data=date;
format life_expectancy_at_birth life_fmt.;
run;
*printare date cu format hipc medie;
title "Printare cu format hipc medie";
proc print data=date;
format hipc hipc_fmt.;
run;



*subset starea de sănătate și riscurile sociale în țările cu șomaj ridicat (peste 10%);
DATA high_unemployment;
    SET date;
    WHERE total_unemployment_rate > 10;
    KEEP country life_expectancy_at_birth mortality_rate_infant current_health_expenditure
         unmet_medical_needs total_unemployment_rate self_perceived_health at_risk_of_poverty_rate;
    FORMAT life_expectancy_at_birth 6.2
           mortality_rate_infant 6.2
           current_health_expenditure 6.2
           unmet_medical_needs 6.2
           total_unemployment_rate 6.2
           self_perceived_health 6.2
           at_risk_of_poverty_rate 6.2;
RUN;
title "Starea de sănătate și riscurile sociale în țările cu șomaj ridicat (peste 10%)"
proc print data=high_unemployment;
run;

*tari cu valori lipsa;
DATA missing_values_subset;
    SET date;
    WHERE missing(country)
          OR missing(life_expectancy_at_birth)
          OR missing(mortality_rate_infant)
          OR missing(current_health_expenditure)
          OR missing(total_air_emissions)
          OR missing(unmet_medical_needs)
          OR missing(total_unemployment_rate)
          OR missing(hipc)
          OR missing(self_perceived_health)
          OR missing(at_risk_of_poverty_rate);
    missing_count = cmiss(of country life_expectancy_at_birth mortality_rate_infant current_health_expenditure
                          total_air_emissions unmet_medical_needs total_unemployment_rate hipc
                          self_perceived_health at_risk_of_poverty_rate);
RUN;

title "Țări cu valori lipsă";
proc print data=missing_values_subset;
    var country missing_count;
run;


*Raport cheltuieli sanatate - pe regiuni + total;
data date_with_region;
    set date;
    length Regiune $20;
    Regiune = put(country, $region_fmt.);
run;
proc sort data=date_with_region;
    by Regiune;
run;

proc means data=date_with_region n sum mean maxdec=2;
    by Regiune;
    var current_health_expenditure;
    title "Raport sumă și medie pentru cheltuieli sănătate";
run;

proc means data=date_with_region n sum mean maxdec=2;
    var current_health_expenditure;
    title "Total general pe toate țările";
run;


* Analiză descriptivă pe toate variabilele numerice;
proc univariate data=date normal plot;
    var life_expectancy_at_birth
        mortality_rate_infant
        current_health_expenditure
        total_air_emissions
        unmet_medical_needs
        total_unemployment_rate
        hipc
        self_perceived_health
        at_risk_of_poverty_rate;
    id country;
    histogram life_expectancy_at_birth
              mortality_rate_infant
              current_health_expenditure
              total_air_emissions
              unmet_medical_needs
              total_unemployment_rate
              hipc
              self_perceived_health
              at_risk_of_poverty_rate;
    title "Statistici descriptive și distribuții: Indicatori de sănătate și economie în Europa";
run;


/* Raport agregat cu media, min, max, număr de valori nenule și lipsă */
data date_regiuni;
    set date;
    regiune = put(country, $region_fmt.);
run;

proc means data=date_regiuni n mean min max nmiss;
    class regiune;
    var life_expectancy_at_birth
        mortality_rate_infant
        current_health_expenditure
        total_air_emissions
        unmet_medical_needs
        total_unemployment_rate
        hipc
        self_perceived_health
        at_risk_of_poverty_rate;
    title 'Indicatori de sănătate și economie pe regiuni europene';
run;

/* interval de încredere pentru media variabilelor socio-economice și de sănătate */
proc means data=date n mean median std clm alpha=0.05 maxdec=2;
    var life_expectancy_at_birth mortality_rate_infant current_health_expenditure
        total_air_emissions unmet_medical_needs total_unemployment_rate hipc
        self_perceived_health at_risk_of_poverty_rate;
    title "intervale de încredere pentru media indicatorilor de sănătate și economie (95%)";
run;


/* frecvente: poverty rate, hipc, unmet med needs*/
data date_cat;
    set date;
    region = put(country, $region_fmt.);

    length hipc_cat $12;
    if hipc < 8 then hipc_cat = 'mic'; *am folosit Q1;
    else if hipc < 12 then hipc_cat = 'mediu'; *am folosit Q3;
    else if hipc ne . then hipc_cat = 'mare';
    else hipc_cat = 'necunoscut';

    length poverty_cat $12;
    if at_risk_of_poverty_rate < 13 then poverty_cat = 'scazut'; *am folosit Q1;
    else if at_risk_of_poverty_rate < 23 then poverty_cat = 'mediu';  *am folosit Q3;
    else if at_risk_of_poverty_rate ne . then poverty_cat = 'ridicat';
    else poverty_cat = 'necunoscut';

    length unmet_med_cat $12;
    if unmet_medical_needs < 0.1 then unmet_med_cat = 'mic';  *am folosit Q1;
    else if unmet_medical_needs < 1 then unmet_med_cat = 'mediu';  *am folosit Q3;
    else if unmet_medical_needs ne . then unmet_med_cat = 'mare';
    else unmet_med_cat = 'necunoscut';

run;


title "Frecvente pentru hipc pe regiuni";
proc freq data=date_cat;
    tables region*hipc_cat / norow nocol nopercent;
run;

title "Frecvente pentru riscul de saracie pe regiuni";
proc freq data=date_cat;
    tables region*poverty_cat / norow nocol nopercent;
run;

title "Frecvente pentru nevoi medicale neindeplinite pe regiuni";
proc freq data=date_cat;
    tables region*unmet_med_cat / norow nocol nopercent;
run;





title 'Distributia HIPC in functie de tara';
goptions reset=all;

* grafic cu bare verticale pentru HIPC pe tara;
proc gchart data=date;
    vbar country / sumvar=hipc type=mean discrete
        coutline=black
        maxis=axis1
        raxis=axis2;
run;


* grafice;
title 'Rata mortalitate infantila pe regiuni';
proc gchart data=date_regiuni;
    vbar regiune / sumvar=mortality_rate_infant type=mean discrete
        coutline=black;
run;
quit;


title 'Persoane aflate in risc de saracie grafic pe tari';
proc gchart data=date;
    HBAR3D country / sumvar=at_risk_of_poverty_rate type=mean discrete
        coutline=black;
run;
quit;


title 'Speranta de viata in Europa de Vest, Nord si Est';
proc gchart data=date_regiuni;
    where regiune in ('Europa de Vest', 'Europa de Nord', 'Europa de Est');
    HBAR3D regiune / sumvar=life_expectancy_at_birth type=mean discrete
        coutline=black;
run;
quit;


title 'Cheltuieli cu sanatatea in Europa de Vest, Nord si Est';
proc gchart data=date_regiuni;
    where regiune in ('Europa de Vest', 'Europa de Nord', 'Europa de Est');
    hbar regiune / sumvar=current_health_expenditure type=mean discrete
        coutline=black;
run;
quit;


title "Distributia ratei somajului";
pattern value=solid;
proc gchart data=date;
    vbar total_unemployment_rate / midpoints=0 to 14 by 2;
run;
quit;



title "Relatia dintre mortalitatea infantila si cheltuieli cu sanatatea";
symbol v=dot i=none c=blue;
proc gplot data=date;
    plot mortality_rate_infant * current_health_expenditure;
run;
quit;


title "Relatia dintre rata riscului de saracie si speranta de viata";
symbol v=dot i=none c=blue;
proc gplot data=date;
    plot life_expectancy_at_birth * at_risk_of_poverty_rate;
run;
quit;




*vom inlocui intai valorile lipsa cu media;
proc stdize data=date reponly method=mean out=date_no_nan;
run;





*analiza de corelatie;
proc corr data=date_no_nan;
   title 'Matricea coeficientilor de corelatie';
run;



*analiza de regresie;


*si le vom standardiza;
proc standard data=date_no_nan mean=0 std=1 out=date_standardizate;
run;

*regresie multipla;
proc reg data=date_standardizate;
    model life_expectancy_at_birth = mortality_rate_infant current_health_expenditure total_air_emissions
                                   unmet_medical_needs total_unemployment_rate hipc self_perceived_health at_risk_of_poverty_rate;
    title "Regresie multipla pentru speranța de viață";
run;
quit;



*anova - speranta de viata;
proc anova data=date;
    class country;
    model life_expectancy_at_birth = country;
    means country / scheffe bon;
    title "ANOVA pentru speranta de viata in functie de tara";
run;
quit;

*sql si sas;
*selectare tari unde speranta viata>75;
proc sql;
  select country, life_expectancy_at_birth
  from date
  where life_expectancy_at_birth > 75
  order by life_expectancy_at_birth desc;
quit;

*5 tari cu cele mai mari cheltuieli pe sanatate;
proc sql outobs=5; *outobs e limit, care nu exista in proc sql;
  select country, current_health_expenditure
  from date
  order by current_health_expenditure desc;
quit;


*calcul indicatori - pe regiuni;
proc print data=date_with_region;

proc sql;
  select
    regiune,
    count(*) as Nr_Tari,
    mean(life_expectancy_at_birth) as Medie_Speranta_Viata format=6.2,
    mean(total_unemployment_rate) as Medie_Somaj format=6.2,
    mean(current_health_expenditure) as Medie_Cheltuieli_Sanatate format=8.2
  from date_with_region
  group by regiune;
quit;



*combinare seturi de date;
data life_exp_data;
infile'/home/u64242353/Life_exp.csv' dsd;
length country $15;
input country $ life_exp;
run;


data inf_mort_data;
infile'/home/u64242353/Infant_mortality_rate.csv' dsd;
length country $15;
input country $ inf_mort;
run;

data at_risk_pov_data;
infile'/home/u64242353/At_risk_poverty.csv' dsd;
length country $15;
input country $ risk_pov;
run;




*fuziune pe baza de corespondenta intre life_exp si inf_mortality;
*intai tb sa sortam dupa var pe care o punem la by;
proc sort data=life_exp_data
out=sorted_life_exp;
by country;
run;

proc sort data=inf_mort_data
out=sorted_infant_mortality;
by country;
run;

data life_mortality_merged;
merge sorted_life_exp sorted_infant_mortality;
by country;
run;

title "Fuziune pe baza de corespondenta - speranta viata si rata de mortalitate infantila";
proc print data=life_mortality_merged;
run;

*facem un set doar cu obs comune;

proc sql;
    create table sql_inner_join_life_pov as
    select
           a.*,
           b.*
    from life_exp_data as a
    inner join at_risk_pov_data as b
        on a.country = b.country;
quit;

title "INNER JOIN: Doar țările comune în ambele tabele";
proc print data=sql_inner_join_life_pov;
run;

proc sql;
    create table sql_left_join as
    select a.country,
           a.*,
           b.*
    from life_exp_data as a
    left join at_risk_pov_data as b
        on a.country = b.country
    where a.life_exp > 70 or b.risk_pov < 20;
quit;

title "LEFT JOIN filtrat: Țări cu speranță de viață > 70 sau risc de sărăcie < 20";
proc print data=sql_left_join (obs=10);
run;

*masive;
data date_missing;
	set date;

	array indicatori[*] life_expectancy_at_birth mortality_rate_infant
	                   current_health_expenditure total_air_emissions
	                   unmet_medical_needs total_unemployment_rate
	                   hipc self_perceived_health at_risk_of_poverty_rate;

	missing_count = 0;
	do i = 1 to dim(indicatori);
		if missing(indicatori[i]) then missing_count + 1;
	end;
run;

proc print data=date_missing;
	where missing_count > 0;
	var country missing_count;
run;

data date_matrix;
	set date;

	array indic[3,3] life_expectancy_at_birth mortality_rate_infant current_health_expenditure
	                 total_air_emissions unmet_medical_needs total_unemployment_rate
	                 hipc self_perceived_health at_risk_of_poverty_rate;

	*sum l1 - life_expectancy_at_birth + mortality_rate_infant + current_health_expenditure;
	*sum l2 - total_air_emissions + unmet_medical_needs + total_unemployment_rate etc;
	array sum_linie[3] sum_l1-sum_l3;
	array sum_coloana[3] sum_c1-sum_c3;

	/* initializez sumele */
	do i = 1 to 3;
		sum_linie[i] = 0;
		sum_coloana[i] = 0;
	end;

	/* calculez suma pe fiecare linie si coloană */
	do i = 1 to 3;
		do j = 1 to 3;
			sum_linie[i] + indic[i,j];
			sum_coloana[j] + indic[i,j];
		end;
	end;
run;

proc print data=date_matrix noobs label;
	var country
	    life_expectancy_at_birth mortality_rate_infant current_health_expenditure
	    total_air_emissions unmet_medical_needs total_unemployment_rate
	    hipc self_perceived_health at_risk_of_poverty_rate
	    sum_l1-sum_l3 sum_c1-sum_c3;
	label sum_l1 = "Suma linie 1"
	      sum_l2 = "Suma linie 2"
	      sum_l3 = "Suma linie 3"
	      sum_c1 = "Suma coloană 1"
	      sum_c2 = "Suma coloană 2"
	      sum_c3 = "Suma coloană 3";
run;


