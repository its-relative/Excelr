DROP TABLE IF EXISTS `books`;
DROP TABLE IF EXISTS `authors`;


CREATE TABLE Authors(AuthorId INTEGER primary key, 
                                Name VARCHAR(70));

insert into Authors values(1,'J K Rowling');
insert into Authors values(2,'Thomas Hardy');
insert into Authors values(3,'Oscar Wilde');
insert into Authors values(4,'Sidney Sheldon');
insert into Authors values(5,'Alistair MacLean');
insert into Authors values(6,'Jane Austen');
insert into Authors values(7,'Chetan Bhagat');
insert into Authors values(8,'Oscar Wilde');

DROP TABLE IF EXISTS `books`;


CREATE TABLE Books(
             BookId INTEGER primary key, 
             Title VARCHAR(50),
             AuthorId INTEGER, foreign key (authorid) references authors(authorid));


insert into Books values(1,'Harry Potter and the Philosopher\'s Stone',1);
insert into Books values(2,'Harry Potter and the Chamber of Secrets',1);
insert into Books values(3,'Harry Potter and the Half-Blood Prince',1);
insert into Books values(4,'Harry Potter and the Goblet of Fire',1);

insert into Books values(5,'Night Without End',5);
insert into Books values(6,'Fear is the Key',5);
insert into Books values(7,'Where Eagles Dare',5);

insert into Books values(8,'Sense and Sensibility',6);
insert into Books values(9,'Pride and Prejudice',6);
insert into Books values(10,'Emma',6);
insert into Books values(11,'Five Point Someone',7);
insert into Books values(12,'Two States',7);
insert into Books values(13,'Salome',8);
insert into Books values(14,'The Happy Prince',8);

DROP TABLE IF EXISTS `drinks`;

CREATE TABLE `drinks` (
  `drinkname` varchar(50) DEFAULT NULL,
  `rate` float(5,2) DEFAULT NULL
) ;


INSERT INTO `drinks` VALUES ('Tea',5.00),('Coffee',15.00),('Pepsi',20.00);



DROP TABLE IF EXISTS `meals`;

CREATE TABLE `meals` (
  `mealname` varchar(50) DEFAULT NULL,
  `rate` float(5,2) DEFAULT NULL
) ;

INSERT INTO `meals` VALUES ('Omlet',10.50),('Sausage',15.50),('Pancake',40.75);

DROP TABLE IF EXISTS `movies`;


create table movies (id integer, title varchar(50), category varchar(25));

insert into movies values(1,	'ASSASSIN\'S CREED: EMBERS',	'Animations'),
(2,	'Real Steel',	'Animations'),
(3,	'Alvin and the Chipmunks',	'Animations'),
(4,	'The Adventures of Tin Tin',	'Animations'),
(5,	'Safe', 	'Action'),
(6,	'Safe House',	'Action'),
(7,	'GIA',	'18+'),
(8,	'Deadline 2009',	'18+'),
(9,	'The Dirty Picture',	'18+'),
(10,	'Marley and me',	'Romance');

DROP TABLE IF EXISTS `members`;

create table members(memid integer, first_name varchar(25), last_name varchar(25), movieid integer);



insert into members values(1,'Alicia','Alarcon', 1),
(2,'Don','Draper', 2),
(3,'Lizzie','Moss', 5),
(4,'Eldon','Chance', 8),
(5,'Jenny','Patterson', 10),
(6,'Craig','Daniels', null),
(7,'Denny', 'Peters',null),
(8,'Patty','Pattinson',null);

DROP TABLE IF EXISTS `person`;

create table person(id integer, fname varchar(20), lname varchar(20));
insert into person values(1,'John','Perry'),(2,'John','Peters'),(3,'John','England'),(4, 'Tom', 'England'),(5,'Tom','Cruise'),(6,'Tom','Cruise'),(7,'John','Peters');


DROP TABLE IF EXISTS `myemp`;

CREATE TABLE `myemp` (
  `EMP_ID` decimal(6,0) NOT NULL DEFAULT 0,
  `FIRST_NAME` varchar(20) DEFAULT NULL,
  `LAST_NAME` varchar(25) NOT NULL,
  `EMAIL` varchar(25) NOT NULL,
  `HIRE_DATE` date NOT NULL,
  `JOB_ID` varchar(10) NOT NULL,
  `SALARY` decimal(8,2) DEFAULT NULL,
  `COMMISSION_PCT` decimal(2,2) DEFAULT NULL,
  `MGR_ID` decimal(6,0) DEFAULT NULL,
  `DEP_ID` decimal(4,0) DEFAULT NULL
) ;


INSERT INTO `myemp` VALUES (100,'Steven','King','SKING','1998-06-17','AD_PRES',24000.00,0.00,0,90),
(101,'Neena','Kochhar','NKOCHHAR','1987-06-18','AD_VP',17000.00,0.00,100,90),
(102,'Lex','De Haan','LDEHAAN','1987-06-19','AD_VP',17000.00,0.00,100,90),
(103,'Alexander','Hunold','AHUNOLD','1976-06-20','IT_PROG',9000.00,0.00,102,60),
(104,'Bruce','Ernst','BERNST','1987-06-21','IT_PROG',6000.00,0.00,103,60),
(105,'David','Austin','DAUSTIN','1987-06-22','IT_PROG',4800.00,0.00,103,60),
(106,'Valli','Pataballa','VPATABAL','1980-06-23','IT_PROG',4800.00,0.00,103,60),
(107,'Diana','Lorentz','DLORENTZ','1987-06-24','IT_PROG',4200.00,0.00,103,60),
(108,'Nancy','Greenberg','NGREENBE','1987-06-25','FI_MGR',12000.00,0.00,101,100),
(109,'Daniel','Faviet','DFAVIET','1992-06-26','FI_ACCOUNT',9000.00,0.00,108,100),
(110,'John','Chen','JCHEN','1987-06-27','FI_ACCOUNT',8200.00,0.00,108,100),
(111,'Ismael','Sciarra','ISCIARRA','1987-06-28','FI_ACCOUNT',7700.00,0.00,108,100),
(112,'Jose Manuel','Urman','JMURMAN','1981-06-29','FI_ACCOUNT',7800.00,0.00,108,100),
(113,'Luis','Popp','LPOPP','1987-06-30','FI_ACCOUNT',6900.00,0.00,108,100),
(114,'Den','Raphaely','DRAPHEAL','1987-07-01','PU_MAN',11000.00,0.00,100,30),
(115,'Alexander','Khoo','AKHOO','1983-07-02','PU_CLERK',3100.00,0.00,114,30),
(116,'Shelli','Baida','SBAIDA','1987-07-03','PU_CLERK',2900.00,0.00,114,30),
(117,'Sigal','Tobias','STOBIAS','1987-07-04','PU_CLERK',2800.00,0.00,114,30),
(118,'Guy','Himuro','GHIMURO','1995-07-05','PU_CLERK',2600.00,0.00,114,30),
(119,'Karen','Colmenares','KCOLMENA','1987-07-06','PU_CLERK',2500.00,0.00,114,30),
(120,'Matthew','Weiss','MWEISS','1987-07-07','ST_MAN',8000.00,0.00,100,50),
(121,'Adam','Fripp','AFRIPP','1990-07-08','ST_MAN',8200.00,0.00,100,50),
(122,'Payam','Kaufling','PKAUFLIN','1987-07-09','ST_MAN',7900.00,0.00,100,50),
(123,'Shanta','Vollman','SVOLLMAN','1987-07-10','ST_MAN',6500.00,0.00,100,50),
(124,'Kevin','Mourgos','KMOURGOS','1997-07-11','ST_MAN',5800.00,0.00,100,50),
(125,'Julia','Nayer','JNAYER','1987-07-12','ST_CLERK',3200.00,0.00,120,50),
(126,'Irene','Mikkilineni','IMIKKILI','1987-07-13','ST_CLERK',2700.00,0.00,120,50),
(127,'James','Landry','JLANDRY','1978-07-14','ST_CLERK',2400.00,0.00,120,50),
(128,'Steven','Markle','SMARKLE','1987-07-15','ST_CLERK',2200.00,0.00,120,50),
(129,'Laura','Bissot','LBISSOT','1987-07-16','ST_CLERK',3300.00,0.00,121,50),
(130,'Mozhe','Atkinson','MATKINSO','1983-07-17','ST_CLERK',2800.00,0.00,121,50),
(131,'James','Marlow','JAMRLOW','1987-07-18','ST_CLERK',2500.00,0.00,121,50),
(132,'TJ','Olson','TJOLSON','1987-07-19','ST_CLERK',2100.00,0.00,121,50),
(133,'Jason','Mallin','JMALLIN','1985-07-20','ST_CLERK',3300.00,0.00,122,50),
(134,'Michael','Rogers','MROGERS','1987-07-21','ST_CLERK',2900.00,0.00,122,50),
(135,'Ki','Gee','KGEE','1987-07-22','ST_CLERK',2400.00,0.00,122,50),
(136,'Hazel','Philtanker','HPHILTAN','1990-07-23','ST_CLERK',2200.00,0.00,122,50),
(137,'Renske','Ladwig','RLADWIG','2000-07-24','ST_CLERK',3600.00,0.00,123,50),
(138,'Stephen','Stiles','SSTILES','1987-07-25','ST_CLERK',3200.00,0.00,123,50),
(139,'John','Seo','JSEO','1989-07-26','ST_CLERK',2700.00,0.00,123,50),
(140,'Joshua','Patel','JPATEL','2011-07-27','ST_CLERK',2500.00,0.00,123,50),
(141,'Trenna','Rajs','TRAJS','1987-07-28','ST_CLERK',3500.00,0.00,124,50),
(142,'Curtis','Davies','CDAVIES','1986-07-29','ST_CLERK',3100.00,0.00,124,50),
(143,'Randall','Matos','RMATOS','1987-07-30','ST_CLERK',2600.00,0.00,124,50),
(144,'Peter','Vargas','PVARGAS','1998-07-31','ST_CLERK',2500.00,0.00,124,50),
(145,'John','Russell','JRUSSEL','1991-08-01','SA_MAN',14000.00,0.40,100,80),
(146,'Karen','Partners','KPARTNER','2019-08-02','SA_MAN',13500.00,0.30,100,80),
(147,'Alberto','Errazuriz','AERRAZUR','1987-08-03','SA_MAN',12000.00,0.30,100,80),
(148,'Gerald','Cambrault','GCAMBRAU','2015-08-04','SA_MAN',11000.00,0.30,100,80),
(149,'Eleni','Zlotkey','EZLOTKEY','1987-08-05','SA_MAN',10500.00,0.20,100,80),
(150,'Peter','Tucker','PTUCKER','1987-08-06','SA_REP',10000.00,0.30,145,80),
(151,'David','Bernstein','DBERNSTE','2013-08-07','SA_REP',9500.00,0.25,145,80),
(152,'Peter','Hall','PHALL','1987-08-08','SA_REP',9000.00,0.25,145,80),
(153,'Christopher','Olsen','COLSEN','1987-08-09','SA_REP',8000.00,0.20,145,80),
(154,'Nanette','Cambrault','NCAMBRAU','1987-08-10','SA_REP',7500.00,0.20,145,80),
(155,'Oliver','Tuvault','OTUVAULT','1987-08-11','SA_REP',7000.00,0.15,145,80),
(156,'Janette','King','JKING','1987-08-12','SA_REP',10000.00,0.35,146,80),
(157,'Patrick','Sully','PSULLY','2002-08-13','SA_REP',9500.00,0.35,146,80),
(158,'Allan','McEwen','AMCEWEN','1987-08-14','SA_REP',9000.00,0.35,146,80),
(159,'Lindsey','Smith','LSMITH','1987-08-15','SA_REP',8000.00,0.30,146,80),
(160,'Louise','Doran','LDORAN','2003-08-16','SA_REP',7500.00,0.30,146,80),
(161,'Sarath','Sewall','SSEWALL','1987-08-17','SA_REP',7000.00,0.25,146,80),
(162,'Clara','Vishney','CVISHNEY','1987-08-18','SA_REP',10500.00,0.25,147,80),
(163,'Danielle','Greene','DGREENE','1987-08-19','SA_REP',9500.00,0.15,147,80),
(164,'Mattea','Marvins','MMARVINS','2007-08-20','SA_REP',7200.00,0.10,147,80),
(165,'David','Lee','DLEE','1987-08-21','SA_REP',6800.00,0.10,147,80),
(166,'Sundar','Ande','SANDE','1987-08-22','SA_REP',6400.00,0.10,147,80),
(167,'Amit','Banda','ABANDA','1987-08-23','SA_REP',6200.00,0.10,147,80),
(168,'Lisa','Ozer','LOZER','1987-08-24','SA_REP',11500.00,0.25,148,80),
(169,'Harrison','Bloom','HBLOOM','1999-08-25','SA_REP',10000.00,0.20,148,80),
(170,'Tayler','Fox','TFOX','1987-08-26','SA_REP',9600.00,0.20,148,80),
(171,'William','Smith','WSMITH','1987-08-27','SA_REP',7400.00,0.15,148,80),
(172,'Elizabeth','Bates','EBATES','1987-08-28','SA_REP',7300.00,0.15,148,80),
(173,'Sundita','Kumar','SKUMAR','1987-08-29','SA_REP',6100.00,0.10,148,80),
(174,'Ellen','Abel','EABEL','1987-08-30','SA_REP',11000.00,0.30,149,80),
(175,'Alyssa','Hutton','AHUTTON','1996-08-31','SA_REP',8800.00,0.25,149,80),
(176,'Jonathon','Taylor','JTAYLOR','1987-09-01','SA_REP',8600.00,0.20,149,80),
(177,'Jack','Livingston','JLIVINGS','1987-09-02','SA_REP',8400.00,0.20,149,80),
(179,'Charles','Johnson','CJOHNSON','1999-09-04','SA_REP',6200.00,0.10,149,80),
(180,'Winston','Taylor','WTAYLOR','1987-09-05','SH_CLERK',3200.00,0.00,120,50),
(181,'Jean','Fleaur','JFLEAUR','1987-09-06','SH_CLERK',3100.00,0.00,120,50),
(182,'Martha','Sullivan','MSULLIVA','1987-09-07','SH_CLERK',2500.00,0.00,120,50),
(183,'Girard','Geoni','GGEONI','1987-09-08','SH_CLERK',2800.00,0.00,120,50),
(184,'Nandita','Sarchand','NSARCHAN','2012-09-09','SH_CLERK',4200.00,0.00,121,50),
(185,'Alexis','Bull','ABULL','1987-09-10','SH_CLERK',4100.00,0.00,121,50),
(186,'Julia','Dellinger','JDELLING','1987-09-11','SH_CLERK',3400.00,0.00,121,50),
(187,'Anthony','Cabrio','ACABRIO','2007-09-12','SH_CLERK',3000.00,0.00,121,50),
(188,'Kelly','Chung','KCHUNG','1987-09-13','SH_CLERK',3800.00,0.00,122,50),
(189,'Jennifer','Dilly','JDILLY','1987-09-14','SH_CLERK',3600.00,0.00,122,50),
(190,'Timothy','Gates','TGATES','1987-09-15','SH_CLERK',2900.00,0.00,122,50),
(191,'Randall','Perkins','RPERKINS','1987-09-16','SH_CLERK',2500.00,0.00,122,50),
(192,'Sarah','Bell','SBELL','1987-09-17','SH_CLERK',4000.00,0.00,123,50),
(193,'Britney','Everett','BEVERETT','1987-09-18','SH_CLERK',3900.00,0.00,123,50),
(194,'Samuel','McCain','SMCCAIN','2000-09-19','SH_CLERK',3200.00,0.00,123,50),
(195,'Vance','Jones','VJONES','1996-09-20','SH_CLERK',2800.00,0.00,123,50),
(196,'Alana','Walsh','AWALSH','1987-09-21','SH_CLERK',3100.00,0.00,124,50),
(197,'Kevin','Feeney','KFEENEY','1987-09-22','SH_CLERK',3000.00,0.00,124,50),
(198,'Donald','OConnell','DOCONNEL','1997-09-23','SH_CLERK',2600.00,0.00,124,50),
(199,'Douglas','Grant','DGRANT','1987-09-24','SH_CLERK',2600.00,0.00,124,50),
(200,'Jennifer','Whalen','JWHALEN','1988-09-25','AD_ASST',4400.00,0.00,101,10),
(201,'Michael','Hartstein','MHARTSTE','1987-09-26','MK_MAN',13000.00,0.00,100,20),
(202,'Pat','Fay','PFAY','1987-09-27','MK_REP',6000.00,0.00,201,20),
(203,'Susan','Mavris','SMAVRIS','1989-09-28','HR_REP',6500.00,0.00,101,40),
(204,'Hermann','Baer','HBAER','1987-09-29','PR_REP',10000.00,0.00,101,70),
(205,'Shelley','Higgins','SHIGGINS','1987-09-30','AC_MGR',12000.00,0.00,101,110),
(206,'William','Gietz','WGIETZ','1997-10-01','AC_ACCOUNT',8300.00,0.00,205,110);

DROP TABLE IF EXISTS `products`;

CREATE TABLE `products` (
  `pcode` varchar(15) DEFAULT NULL,
  `pname` varchar(70) DEFAULT NULL,
  `productLine` varchar(50) NOT NULL,
  `pvendor` varchar(50) DEFAULT NULL,
  `qtyInStock` smallint(6) DEFAULT NULL,
  `MRP` decimal(7,2) DEFAULT NULL
) ;

INSERT INTO `products` VALUES ('S10_1678','1969 Harley Davidson Ultimate Chopper','Motorcycles','Min Lin Diecast',7933,95.70),('S10_1949','1952 Alpine Renault 1300','Classic Cars','Classic Metal Creations',7305,214.30),('S10_2016','1996 Moto Guzzi 1100i','Motorcycles','Highway 66 Mini Classics',6625,118.94),('S10_4698','2003 Harley-Davidson Eagle Drag Bike','Motorcycles','Red Start Diecast',5582,193.66),('S10_4757','1972 Alfa Romeo GTA','Classic Cars','Motor City Art Classics',3252,136.00),('S10_4962','1962 LanciaA Delta 16V','Classic Cars','Second Gear Diecast',6791,147.74),('S12_1099','1968 Ford Mustang','Classic Cars','Autoart Studio Design',68,194.57),('S12_1108','2001 Ferrari Enzo','Classic Cars','Second Gear Diecast',3619,207.80),('S12_1666','1958 Setra Bus','Trucks and Buses','Welly Diecast Productions',1579,136.67),('S12_2823','2002 Suzuki XREO','Motorcycles','Unimax Art Galleries',9997,150.62),('S12_3148','1969 Corvair Monza','Classic Cars','Welly Diecast Productions',6906,151.08),('S12_3380','1968 Dodge Charger','Classic Cars','Welly Diecast Productions',9123,117.44),('S12_3891','1969 Ford Falcon','Classic Cars','Second Gear Diecast',1049,173.02),('S12_3990','1970 Plymouth Hemi Cuda','Classic Cars','Studio M Art Models',5663,79.80),('S12_4473','1957 Chevy Pickup','Trucks and Buses','Exoto Designs',6125,118.50),('S12_4675','1969 Dodge Charger','Classic Cars','Welly Diecast Productions',7323,115.16),('S18_1097','1940 Ford Pickup Truck','Trucks and Buses','Studio M Art Models',2613,116.67),('S18_1129','1993 Mazda RX-7','Classic Cars','Highway 66 Mini Classics',3975,141.54),('S18_1342','1937 Lincoln Berline','Vintage Cars','Motor City Art Classics',8693,102.74),('S18_1367','1936 Mercedes-Benz 500K Special Roadster','Vintage Cars','Studio M Art Models',8635,53.91),('S18_1589','1965 Aston Martin DB5','Classic Cars','Classic Metal Creations',9042,124.44),('S18_1662','1980s Black Hawk Helicopter','Planes','Red Start Diecast',5330,157.69),('S18_1749','1917 Grand Touring Sedan','Vintage Cars','Welly Diecast Productions',2724,170.00),('S18_1889','1948 Porsche 356-A Roadster','Classic Cars','Gearbox Collectibles',8826,77.00),('S18_1984','1995 Honda Civic','Classic Cars','Min Lin Diecast',9772,142.25),('S18_2238','1998 Chrysler Plymouth Prowler','Classic Cars','Gearbox Collectibles',4724,163.73),('S18_2248','1911 Ford Town Car','Vintage Cars','Motor City Art Classics',540,60.54),('S18_2319','1964 Mercedes Tour Bus','Trucks and Buses','Unimax Art Galleries',8258,122.73),('S18_2325','1932 Model A Ford J-Coupe','Vintage Cars','Autoart Studio Design',9354,127.13),('S18_2432','1926 Ford Fire Engine','Trucks and Buses','Carousel DieCast Legends',2018,60.77),('S18_2581','P-51-D Mustang','Planes','Gearbox Collectibles',992,84.48),('S18_2625','1936 Harley Davidson El Knucklehead','Motorcycles','Welly Diecast Productions',4357,60.57),('S18_2795','1928 Mercedes-Benz SSK','Vintage Cars','Gearbox Collectibles',548,168.75),('S18_2870','1999 Indy 500 Monte Carlo SS','Classic Cars','Red Start Diecast',8164,132.00),('S18_2949','1913 Ford Model T Speedster','Vintage Cars','Carousel DieCast Legends',4189,101.31),('S18_2957','1934 Ford V8 Coupe','Vintage Cars','Min Lin Diecast',5649,62.46),('S18_3029','1999 Yamaha Speed Boat','Ships','Min Lin Diecast',4259,86.02),('S18_3136','18th Century Vintage Horse Carriage','Vintage Cars','Red Start Diecast',5992,104.72),('S18_3140','1903 Ford Model A','Vintage Cars','Unimax Art Galleries',3913,136.59),('S18_3232','1992 Ferrari 360 Spider red','Classic Cars','Unimax Art Galleries',8347,169.34),('S18_3233','1985 Toyota Supra','Classic Cars','Highway 66 Mini Classics',7733,107.57),('S18_3259','Collectable Wooden Train','Trains','Carousel DieCast Legends',6450,100.84),('S18_3278','1969 Dodge Super Bee','Classic Cars','Min Lin Diecast',1917,80.41),('S18_3320','1917 Maxwell Touring Car','Vintage Cars','Exoto Designs',7913,99.21),('S18_3482','1976 Ford Gran Torino','Classic Cars','Gearbox Collectibles',9127,146.99),('S18_3685','1948 Porsche Type 356 Roadster','Classic Cars','Gearbox Collectibles',8990,141.28),('S18_3782','1957 Vespa GS150','Motorcycles','Studio M Art Models',7689,62.17),('S18_3856','1941 Chevrolet Special Deluxe Cabriolet','Vintage Cars','Exoto Designs',2378,105.87),('S18_4027','1970 Triumph Spitfire','Classic Cars','Min Lin Diecast',5545,143.62),('S18_4409','1932 Alfa Romeo 8C2300 Spider Sport','Vintage Cars','Exoto Designs',6553,92.03),('S18_4522','1904 Buick Runabout','Vintage Cars','Exoto Designs',8290,87.77),('S18_4600','1940s Ford truck','Trucks and Buses','Motor City Art Classics',3128,121.08),('S18_4668','1939 Cadillac Limousine','Vintage Cars','Studio M Art Models',6645,50.31),('S18_4721','1957 Corvette Convertible','Classic Cars','Classic Metal Creations',1249,148.80),('S18_4933','1957 Ford Thunderbird','Classic Cars','Studio M Art Models',3209,71.27),('S24_1046','1970 Chevy Chevelle SS 454','Classic Cars','Unimax Art Galleries',1005,73.49),('S24_1444','1970 Dodge Coronet','Classic Cars','Highway 66 Mini Classics',4074,57.80),('S24_1578','1997 BMW R 1100 S','Motorcycles','Autoart Studio Design',7003,112.70),('S24_1628','1966 Shelby Cobra 427 S/C','Classic Cars','Carousel DieCast Legends',8197,50.31),('S24_1785','1928 British Royal Navy Airplane','Planes','Classic Metal Creations',3627,109.42),('S24_1937','1939 Chevrolet Deluxe Coupe','Vintage Cars','Motor City Art Classics',7332,33.19),('S24_2000','1960 BSA Gold Star DBD34','Motorcycles','Highway 66 Mini Classics',15,76.17),('S24_2011','18th century schooner','Ships','Carousel DieCast Legends',1898,122.89),('S24_2022','1938 Cadillac V-16 Presidential Limousine','Vintage Cars','Classic Metal Creations',2847,44.80),('S24_2300','1962 Volkswagen Microbus','Trucks and Buses','Autoart Studio Design',2327,127.79),('S24_2360','1982 Ducati 900 Monster','Motorcycles','Highway 66 Mini Classics',6840,69.26),('S24_2766','1949 Jaguar XK 120','Classic Cars','Classic Metal Creations',2350,90.87),('S24_2840','1958 Chevy Corvette Limited Edition','Classic Cars','Carousel DieCast Legends',2542,35.36),('S24_2841','1900s Vintage Bi-Plane','Planes','Autoart Studio Design',5942,68.51),('S24_2887','1952 Citroen-15CV','Classic Cars','Exoto Designs',1452,117.44),('S24_2972','1982 Lamborghini Diablo','Classic Cars','Second Gear Diecast',7723,37.76),('S24_3151','1912 Ford Model T Delivery Wagon','Vintage Cars','Min Lin Diecast',9173,88.51),('S24_3191','1969 Chevrolet Camaro Z28','Classic Cars','Exoto Designs',4695,85.61),('S24_3371','1971 Alpine Renault 1600s','Classic Cars','Welly Diecast Productions',7995,61.23),('S24_3420','1937 Horch 930V Limousine','Vintage Cars','Autoart Studio Design',2902,65.75),('S24_3432','2002 Chevy Corvette','Classic Cars','Gearbox Collectibles',9446,107.08),('S24_3816','1940 Ford Delivery Sedan','Vintage Cars','Carousel DieCast Legends',6621,83.86),('S24_3856','1956 Porsche 356A Coupe','Classic Cars','Classic Metal Creations',6600,140.43),('S24_3949','Corsair F4U ( Bird Cage)','Planes','Second Gear Diecast',6812,68.24),('S24_3969','1936 Mercedes Benz 500k Roadster','Vintage Cars','Red Start Diecast',2081,41.03),('S24_4048','1992 Porsche Cayenne Turbo Silver','Classic Cars','Exoto Designs',6582,118.28),('S24_4258','1936 Chrysler Airflow','Vintage Cars','Second Gear Diecast',4710,97.39),('S24_4278','1900s Vintage Tri-Plane','Planes','Unimax Art Galleries',2756,72.45),('S24_4620','1961 Chevrolet Impala','Classic Cars','Classic Metal Creations',7869,80.84),('S32_1268','1980’s GM Manhattan Express','Trucks and Buses','Motor City Art Classics',5099,96.31),('S32_1374','1997 BMW F650 ST','Motorcycles','Exoto Designs',178,99.89),('S32_2206','1982 Ducati 996 R','Motorcycles','Gearbox Collectibles',9241,40.23),('S32_2509','1954 Greyhound Scenicruiser','Trucks and Buses','Classic Metal Creations',2874,54.11),('S32_3207','1950\'s Chicago Surface Lines Streetcar','Trains','Gearbox Collectibles',8601,62.14),('S32_3522','1996 Peterbilt 379 Stake Bed with Outrigger','Trucks and Buses','Red Start Diecast',814,64.64),('S32_4289','1928 Ford Phaeton Deluxe','Vintage Cars','Highway 66 Mini Classics',136,68.79),('S32_4485','1974 Ducati 350 Mk3 Desmo','Motorcycles','Second Gear Diecast',3341,102.05),('S50_1341','1930 Buick Marquette Phaeton','Vintage Cars','Studio M Art Models',7062,43.64),('S50_1392','Diamond T620 Semi-Skirted Tanker','Trucks and Buses','Highway 66 Mini Classics',1016,115.75),('S50_1514','1962 City of Detroit Streetcar','Trains','Classic Metal Creations',1645,58.58),('S50_4713','2002 Yamaha YZR M1','Motorcycles','Autoart Studio Design',600,81.36),('S700_1138','The Schooner Bluenose','Ships','Autoart Studio Design',1897,66.67),('S700_1691','American Airlines: B767-300','Planes','Min Lin Diecast',5841,91.34),('S700_1938','The Mayflower','Ships','Studio M Art Models',737,86.61),('S700_2047','HMS Bounty','Ships','Unimax Art Galleries',3501,90.52),('S700_2466','America West Airlines B757-200','Planes','Motor City Art Classics',9653,99.72),('S700_2610','The USS Constitution Ship','Ships','Red Start Diecast',7083,72.28),('S700_2824','1982 Camaro Z28','Classic Cars','Carousel DieCast Legends',6934,101.15),('S700_2834','ATA: B757-300','Planes','Highway 66 Mini Classics',7106,118.65),('S700_3167','F/A 18 Hornet 1/72','Planes','Motor City Art Classics',551,80.00),('S700_3505','The Titanic','Ships','Carousel DieCast Legends',1956,100.17),('S700_3962','The Queen Mary','Ships','Welly Diecast Productions',5088,99.31),('S700_4002','American Airlines: MD-11S','Planes','Second Gear Diecast',8820,74.03),('S72_1253','Boeing X-32A JSF','Planes','Motor City Art Classics',4857,49.66),('S72_3212','Pont Yacht','Ships','Unimax Art Galleries',414,54.60);

DROP TABLE IF EXISTS `orders`;

CREATE TABLE `orders` (
  `orderid` int(11) NOT NULL AUTO_INCREMENT,
  `product_code` varchar(15) DEFAULT NULL,
  `vendor` varchar(50) DEFAULT NULL,
  `status` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`orderid`)
) ;

DROP TABLE IF EXISTS `fraud`;

CREATE TABLE `fraud` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `client_id` int(11) DEFAULT NULL,
  `message` varchar(50) DEFAULT NULL,
  `tot` datetime DEFAULT NULL,
  PRIMARY KEY (`id`)
) ;


DROP TABLE IF EXISTS `payments`;


CREATE TABLE `payments` (
  `client_id` int(11) DEFAULT NULL,
  `amount` int(11) NOT NULL
) ;


DROP TABLE IF EXISTS `book_sales`;


CREATE TABLE `book_sales` (
  `sale_id` int(11) NOT NULL AUTO_INCREMENT,
  `bookid` int(11) DEFAULT NULL,
  `title` varchar(100) DEFAULT NULL,
  `qty_sold` int(11) DEFAULT NULL,
  `tos` datetime DEFAULT NULL,
  PRIMARY KEY (`sale_id`)
) ;




DROP TABLE IF EXISTS `employee`;
DROP TABLE IF EXISTS `offices`;

CREATE TABLE `offices` (
  `officeCode` varchar(10) NOT NULL,
  `city` varchar(50) NOT NULL,
  `phone` varchar(50) NOT NULL,
  `addressLine1` varchar(50) NOT NULL,
  `addressLine2` varchar(50) DEFAULT NULL,
  `state` varchar(50) DEFAULT NULL,
  `country` varchar(50) NOT NULL,
  `postalCode` varchar(15) NOT NULL,
  `territory` varchar(10) NOT NULL,
  PRIMARY KEY (`officeCode`)
) ;

INSERT INTO `offices` VALUES ('1','San Francisco','+1 650 219 4782','100 Market Street','Suite 300','CA','USA','94080','NA'),('2','Boston','+1 215 837 0825','1550 Court Place','Suite 102','MA','USA','02107','NA'),('3','NYC','+1 212 555 3000','523 East 53rd Street','apt. 5A','NY','USA','10022','NA'),('4','Paris','+33 14 723 4404','43 Rue Jouffroy D\'abbans',NULL,NULL,'France','75017','EMEA'),('5','Tokyo','+81 33 224 5000','4-1 Kioicho',NULL,'Chiyoda-Ku','Japan','102-8578','Japan'),('6','Sydney','+61 2 9264 2451','5-11 Wentworth Avenue','Floor #2',NULL,'Australia','NSW 2010','APAC'),('7','London','+44 20 7877 2041','25 Old Broad Street','Level 7',NULL,'UK','EC2N 1HN','EMEA');




CREATE TABLE `employee` (
  `employeeNumber` int NOT NULL,
  `lastName` varchar(50) NOT NULL,
  `firstName` varchar(50) NOT NULL,
  `extension` varchar(10) NOT NULL,
  `email` varchar(100) NOT NULL,
  `officeCode` varchar(10) NOT NULL,
  `reportsTo` int DEFAULT NULL,
  `jobTitle` varchar(50) NOT NULL,
  PRIMARY KEY (`employeeNumber`),
  KEY `reportsTo` (`reportsTo`),
  KEY `officeCode` (`officeCode`),
  CONSTRAINT `employees_ibfk_1` FOREIGN KEY (`reportsTo`) REFERENCES `employee` (`employeeNumber`),
  CONSTRAINT `employees_ibfk_2` FOREIGN KEY (`officeCode`) REFERENCES `offices` (`officeCode`)
) ;

INSERT INTO `employee` VALUES (1002,'Murphy','Diane','x5800','dmurphy@classicmodelcars.com','1',NULL,'President'),(1056,'Patterson','Mary','x4611','mpatterso@classicmodelcars.com','1',1002,'VP Sales'),(1076,'Firrelli','Jeff','x9273','jfirrelli@classicmodelcars.com','1',1002,'VP Marketing'),(1088,'Patterson','William','x4871','wpatterson@classicmodelcars.com','6',1056,'Sales Manager (APAC)'),(1102,'Bondur','Gerard','x5408','gbondur@classicmodelcars.com','4',1056,'Sale Manager (EMEA)'),(1143,'Bow','Anthony','x5428','abow@classicmodelcars.com','1',1056,'Sales Manager (NA)'),(1165,'Jennings','Leslie','x3291','ljennings@classicmodelcars.com','1',1143,'Sales Rep'),(1166,'Thompson','Leslie','x4065','lthompson@classicmodelcars.com','1',1143,'Sales Rep'),(1188,'Firrelli','Julie','x2173','jfirrelli@classicmodelcars.com','2',1143,'Sales Rep'),(1216,'Patterson','Steve','x4334','spatterson@classicmodelcars.com','2',1143,'Sales Rep'),(1286,'Tseng','Foon Yue','x2248','ftseng@classicmodelcars.com','3',1143,'Sales Rep'),(1323,'Vanauf','George','x4102','gvanauf@classicmodelcars.com','3',1143,'Sales Rep'),(1337,'Bondur','Loui','x6493','lbondur@classicmodelcars.com','4',1102,'Sales Rep'),(1370,'Hernandez','Gerard','x2028','ghernande@classicmodelcars.com','4',1102,'Sales Rep'),(1401,'Castillo','Pamela','x2759','pcastillo@classicmodelcars.com','4',1102,'Sales Rep'),(1501,'Bott','Larry','x2311','lbott@classicmodelcars.com','7',1102,'Sales Rep'),(1504,'Jones','Barry','x102','bjones@classicmodelcars.com','7',1102,'Sales Rep'),(1611,'Fixter','Andy','x101','afixter@classicmodelcars.com','6',1088,'Sales Rep'),(1612,'Marsh','Peter','x102','pmarsh@classicmodelcars.com','6',1088,'Sales Rep'),(1619,'King','Tom','x103','tking@classicmodelcars.com','6',1088,'Sales Rep'),(1621,'Nishi','Mami','x101','mnishi@classicmodelcars.com','5',1056,'Sales Rep'),(1625,'Kato','Yoshimi','x102','ykato@classicmodelcars.com','5',1621,'Sales Rep'),(1702,'Gerard','Martin','x2312','mgerard@classicmodelcars.com','4',1102,'Sales Rep');


drop table if exists sales;

CREATE TABLE sales(
    sales_employee VARCHAR(50) NOT NULL,
    fiscal_year INT NOT NULL,
    sale DECIMAL(14,2) NOT NULL,
    PRIMARY KEY(sales_employee,fiscal_year)
);

INSERT INTO sales(sales_employee,fiscal_year,sale)
VALUES('Bob',2016,100),
      ('Bob',2017,150),
      ('Bob',2018,200),
      ('Alice',2016,150),
      ('Alice',2017,100),
      ('Alice',2018,200),
       ('John',2016,200),
      ('John',2017,150),
      ('John',2018,250);

drop table if exists t ;
CREATE TABLE t (
    val INT
);

INSERT INTO t(val)
VALUES(1),(2),(2),(3),(4),(4),(5);


drop table if exists trains;
create table trains(Train_id integer,   Station varchar(50),  Time time);
insert into trains values (110,'San Francisco', '10:00:00'),
(110,'Redwood City',  '10:54:00'),
(110,'Palo Alto','11:02:00'),
(110 ,'San Jose', '12:35:00'),
(120,'San Francisco','11:00:00'),
(120 ,'Palo Alto', '12:49:00'),
(120,'San Jose', '13:30:00');

drop table if exists students_d;
CREATE TABLE `students_d` (
  `date_id` date DEFAULT NULL,
  `student_id` int DEFAULT NULL,
  `attendance` varchar(10) DEFAULT NULL
);

insert into students_d values('2022-02-14' ,     1,'p'),
('2022-02-14' ,     2,'p'),
('2022-02-14' ,     3,'a'),
('2022-02-14' ,     4,'p'),
('2022-02-15' ,     1,'p'),
('2022-02-15' ,     2,'p'),
('2022-02-15' ,     3,'a'),
('2022-02-15' ,     4,'p'),
('2022-02-16' ,     1,'p'),
('2022-02-16' ,     2,'p'),
('2022-02-16' ,     3,'a'),
('2022-02-16' ,     4,'p'),
('2022-02-17' ,     1,'p'),
('2022-02-17' ,     2,'a'),
('2022-02-17' ,    3,'p'),
('2022-02-17' ,     4,'p');



drop table if exists basic_pays;
CREATE TABLE basic_pays (
  employee_id int,
  fiscal_year INT,
  salary DECIMAL(10 , 2 ),
  PRIMARY KEY (employee_id, fiscal_year)
);


INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(100,2017,24000);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(101,2017,17000);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(102,2017,17000);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(103,2017,9000);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(104,2017,6000);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(105,2017,4800);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(106,2017,4800);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(107,2017,4200);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(108,2017,12000);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(109,2017,9000);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(110,2017,8200);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(111,2017,7700);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(112,2017,7800);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(113,2017,6900);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(114,2017,11000);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(115,2017,3100);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(116,2017,2900);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(117,2017,2800);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(118,2017,2600);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(119,2017,2500);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(120,2017,8000);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(121,2017,8200);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(122,2017,7900);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(123,2017,6500);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(126,2017,2700);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(145,2017,14000);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(146,2017,13500);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(176,2017,8600);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(177,2017,8400);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(178,2017,7000);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(179,2017,6200);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(192,2017,4000);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(193,2017,3900);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(200,2017,4400);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(201,2017,13000);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(202,2017,6000);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(203,2017,6500);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(204,2017,10000);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(205,2017,12000);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(206,2017,8300);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(100,2018,25920);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(101,2018,18190);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(102,2018,18360);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(103,2018,9720);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(104,2018,6060);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(105,2018,4992);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(106,2018,5040);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(107,2018,4284);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(108,2018,12360);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(109,2018,9540);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(110,2018,8692);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(111,2018,7931);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(112,2018,8580);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(113,2018,7107);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(114,2018,11440);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(115,2018,3131);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(116,2018,3161);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(117,2018,2940);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(118,2018,2652);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(119,2018,2650);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(120,2018,8800);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(121,2018,8364);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(122,2018,8611);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(123,2018,6565);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(126,2018,2808);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(145,2018,14560);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(146,2018,14580);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(176,2018,9202);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(177,2018,8988);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(178,2018,7630);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(179,2018,6448);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(192,2018,4320);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(193,2018,4173);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(200,2018,4620);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(201,2018,13000);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(202,2018,6360);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(203,2018,7085);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(204,2018,10100);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(205,2018,12360);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(206,2018,8632);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(100,2020,26179.2);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(101,2020,19463.3);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(102,2020,19278);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(103,2020,10206);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(104,2020,6605.4);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(105,2020,5391.36);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(106,2020,5191.2);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(107,2020,4498.2);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(108,2020,13472.4);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(109,2020,9826.2);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(110,2020,9561.2);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(111,2020,8248.24);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(112,2020,9352.2);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(113,2020,7107);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(114,2020,12012);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(115,2020,3224.93);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(116,2020,3287.44);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(117,2020,3175.2);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(118,2020,2864.16);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(119,2020,2782.5);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(120,2020,9152);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(121,2020,8531.28);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(122,2020,8697.11);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(123,2020,6630.65);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(126,2020,2920.32);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(145,2020,16016);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(146,2020,14871.6);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(176,2020,9938.16);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(177,2020,9167.76);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(178,2020,7858.9);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(179,2020,6641.44);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(192,2020,4406.4);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(193,2020,4339.92);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(200,2020,4712.4);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(201,2020,14040);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(202,2020,6614.4);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(203,2020,7155.85);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(204,2020,10908);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(205,2020,12730.8);
INSERT INTO basic_pays(employee_id, fiscal_year,salary) VALUES(206,2020,8890.96);


drop table if exists cats;

CREATE TABLE `cats` (
  `name` text,
  `breed` text,
  `weight` double DEFAULT NULL,
  `color` text,
  `age` int DEFAULT NULL
) ;

INSERT INTO `cats` VALUES ('Ashes','Persian',4.5,'Black',5),('Molly','Persian',4.2,'Black',1),('Felix','Persian',5,'Tortoiseshell',2),('Smudge','British Shorthair',4.9,'Black',4),('Tigger','British Shorthair',3.8,'Tortoiseshell',2),('Alfie','Siamese',5.5,'Brown',5),('Oscar','Siamese',6.1,'Black',1),('Millie','Maine Coon',5.4,'Tortoiseshell',5),('Misty','Maine Coon',5.7,'Brown',2),('Puss','Maine Coon',5.1,'Tortoiseshell',2),('Smokey','Maine Coon',6.1,'Brown',4),('Charlie','British Shorthair',4.8,'Black',4);

drop table if exists skills;
create table skills(cadidate_id integer, skill varchar(25));
insert into skills values(123 ,'Python'),
(123  ,'Tableau'),
(123  ,'PostgreSQL'),
(234  ,'R'), (234 ,'PowerBI'),
(234  ,'SQL Server'),
(345  ,'Python'),
(345  ,'Tableau'),
(147  ,'Python'),
(147  ,'PostgreSQL'),
(147  ,'Tableau'),
(147  ,'Java'),
(168  ,'Python'),
(256  ,'Tableau');

-- Subqueries
drop table if exists mycust;

CREATE TABLE mycust(  
  cust_id int NOT NULL,  
  name varchar(35),  
  occupation varchar(25),  
  age int  
);  

INSERT INTO mycust(cust_id, name, occupation, age)   
VALUES (101, 'Peter', 'Engineer', 32),  
(102, 'Joseph', 'Developer', 30),  
(103, 'John', 'Leader', 28),  
(104, 'Stephen', 'Scientist', 45),  
(105, 'Suzi', 'Carpenter', 26),  
(106, 'Bob', 'Actor', 25),  
(107, NULL, NULL, NULL);  


drop table if exists myord;

CREATE TABLE myord (  
    order_id int NOT NULL,   
    cust_id int,   
    prod_name varchar(45),  
    order_date date  
);  

INSERT INTO myord (order_id, cust_id, prod_name, order_date)   
VALUES (1, '101', 'Laptop', '2020-01-10'),  
(2, '103', 'Desktop', '2020-02-12'),  
(3, '106', 'Iphone', '2020-02-15'),  
(4, '104', 'Mobile', '2020-03-05'),  
(5, '102', 'TV', '2020-03-20');  

drop table if exists marks;
create table marks(sid integer primary key, sname varchar(30), mark integer);

insert into marks values(1,'Tom',45),(2,'Tim',87),(3,'Jim',56),(4,'Jon',78),(5,'Harry',52),(6,'Potter',59);


drop table if exists w_orders;
  CREATE TABLE w_orders (
  order_id INT,
  order_date DATE,
  order_total INT
);

INSERT INTO w_orders (order_id, order_date, order_total) VALUES
(1, '2020-04-03', 100),
(2, '2020-04-03', 250),
(3, '2020-04-04', 80),
(4, '2020-04-05', 10),
(5, '2020-04-03', 120),
(6, '2020-04-04', 90),
(7, '2020-04-04', 50),
(8, '2020-04-04', 15);



DROP TABLE IF EXISTS `uber_logs`;

CREATE TABLE `uber_logs` (
  `request_id` int DEFAULT NULL,
  `request_date` text,
  `request_status` text,
  `distance_to_travel` double DEFAULT NULL,
  `monetary_cost` double DEFAULT NULL,
  `driver_to_client_distance` double DEFAULT NULL
) ;

DROP TABLE IF EXISTS country;

-- Table: country
CREATE TABLE country (
    id int  PRIMARY KEY auto_increment,
    country_name varchar(128)  NOT NULL,
    country_name_eng varchar(128)  NOT NULL,
    country_code char(8)  NOT NULL
);

DROP TABLE IF EXISTS city;

-- Table: city
CREATE TABLE city (
    id int  PRIMARY KEY,
    city_name varchar(128)  NOT NULL,
    latitude float(9,6)  NOT NULL,
    longitude float(9,6)  NOT NULL,
    country_id int  NOT NULL
);

-- Insert Values
INSERT INTO country  VALUES (1,'Deutschland', 'Germany', 'DEU');
INSERT INTO country  VALUES (2,'Srbija', 'Serbia', 'SRB');
INSERT INTO country VALUES (3,'Hrvatska', 'Croatia', 'HRV');
INSERT INTO country VALUES (4,'United Stated of America', 'United Stated of America', 'USA');
INSERT INTO country VALUES (5,'Polska', 'Poland', 'POL');

INSERT INTO city VALUES (1,'Berlin', 52.520008, 13.404954, 1);
INSERT INTO city  VALUES (2,'Belgrade', 44.787197, 20.457273, 2);
INSERT INTO city VALUES (3,'Zagreb', 45.815399, 15.966568, 3);
INSERT INTO city VALUES (4,'New York', 40.73061, -73.935242, 4);
INSERT INTO city VALUES (5,'Los Angeles', 34.052235, -118.243683, 4);
INSERT INTO city  VALUES (6,'Warsaw', 52.237049, 21.017532, 5);

DROP TABLE IF EXISTS customer;
-- Table: customer
CREATE TABLE customer (
    id int  PRIMARY KEY AUTO_INCREMENT,
    customer_name varchar(255)  NOT NULL,
    city_id int  NOT NULL,
    customer_address varchar(255)  NOT NULL,
    next_call_date date  NULL,
    ts_inserted datetime  NOT NULL
);

INSERT INTO customer (customer_name, city_id, customer_address, next_call_date, ts_inserted) VALUES ('Jewelry Store', 4, 'Long Street 120', '2020/1/21', '2020/1/9 14:1:20');
INSERT INTO customer (customer_name, city_id, customer_address, next_call_date, ts_inserted) VALUES ('Bakery', 1, 'Kurfürstendamm 25', '2020/2/21', '2020/1/9 17:52:15');
INSERT INTO customer (customer_name, city_id, customer_address, next_call_date, ts_inserted) VALUES ('Café', 1, 'Tauentzienstraße 44', '2020/1/21', '2020/1/10 8:2:49');
INSERT INTO customer (customer_name, city_id, customer_address, next_call_date, ts_inserted) VALUES ('Restaurant', 3, 'Ulica lipa 15', '2020/1/21', '2020/1/10 9:20:21');



-- DROP TABLE IF EXISTS employee;
-- Table: employee
/* CREATE TABLE employee (
    id int  PRIMARY KEY AUTO_INCREMENT,
    first_name varchar(255)  NOT NULL,
    last_name varchar(255)  NOT NULL
);

INSERT INTO employee (first_name, last_name) VALUES ('Thomas (Neo)', 'Anderson');
INSERT INTO employee (first_name, last_name) VALUES ('Agent', 'Smith'); */


DROP TABLE IF EXISTS calls;

CREATE TABLE calls (
    id int  PRIMARY KEY AUTO_INCREMENT,
    employee_id int  NOT NULL,
    customer_id int  NOT NULL,
    start_time datetime  NOT NULL,
    end_time datetime  NULL,
    call_outcome_id int  NULL
);

INSERT INTO calls (employee_id, customer_id, start_time, end_time, call_outcome_id) VALUES (1, 4, '2020/1/11 9:0:15', '2020/1/11 9:12:22', 2);
INSERT INTO calls (employee_id, customer_id, start_time, end_time, call_outcome_id) VALUES (1, 2, '2020/1/11 9:14:50', '2020/1/11 9:20:1', 2);
INSERT INTO calls (employee_id, customer_id, start_time, end_time, call_outcome_id) VALUES (2, 3, '2020/1/11 9:2:20', '2020/1/11 9:18:5', 3);
INSERT INTO calls (employee_id, customer_id, start_time, end_time, call_outcome_id) VALUES (1, 1, '2020/1/11 9:24:15', '2020/1/11 9:25:5', 3);
INSERT INTO calls (employee_id, customer_id, start_time, end_time, call_outcome_id) VALUES (1, 3, '2020/1/11 9:26:23', '2020/1/11 9:33:45', 2);
INSERT INTO calls (employee_id, customer_id, start_time, end_time, call_outcome_id) VALUES (1, 2, '2020/1/11 9:40:31', '2020/1/11 9:42:32', 2);
INSERT INTO calls (employee_id, customer_id, start_time, end_time, call_outcome_id) VALUES (2, 4, '2020/1/11 9:41:17', '2020/1/11 9:45:21', 2);
INSERT INTO calls (employee_id, customer_id, start_time, end_time, call_outcome_id) VALUES (1, 1, '2020/1/11 9:42:32', '2020/1/11 9:46:53', 3);
INSERT INTO calls (employee_id, customer_id, start_time, end_time, call_outcome_id) VALUES (2, 1, '2020/1/11 9:46:0', '2020/1/11 9:48:2', 2);
INSERT INTO calls (employee_id, customer_id, start_time, end_time, call_outcome_id) VALUES (2, 2, '2020/1/11 9:50:12', '2020/1/11 9:55:35', 2);


DROP TABLE IF EXISTS call_outcome;

-- Table: call_outcome
CREATE TABLE call_outcome (
    id int  PRIMARY KEY AUTO_INCREMENT,
    outcome_text char(128)  NOT NULL
);

-- insert values
INSERT INTO call_outcome (outcome_text) VALUES ('call started');
INSERT INTO call_outcome (outcome_text) VALUES ('finished - successfully');
INSERT INTO call_outcome (outcome_text) VALUES ('finished - unsuccessfully');

-- Table Parents

drop table if exists parents;

create table parents(Id integer, Name varchar(20), Age integer,ParentId integer);

insert into parents values(1,'Jonathan',5,3),
(2,'Alexandra',7,3),
(3,'Barbara',30,4),
(4,'Tom',50,6),
(5,'George',55,6),
(6,'Amy',80,7),
(7,'Josephine',99,35);

drop table if exists gdp;
create table gdp(fiscal_year year(4), gdp integer);
insert into gdp values('1990',150),('1991',130),('1992',140),('1993',138),('1994',142),('1995',128),('1996',210),('1997',213);

--
-- Table structure for table stock
--

DROP TABLE IF EXISTS stock;

CREATE TABLE stock (
  date date DEFAULT NULL,
  ticker text,
  open double DEFAULT NULL,
  high double DEFAULT NULL,
  low double DEFAULT NULL,
  close double DEFAULT NULL
) ;

--
-- Add records to table stock
--


INSERT INTO stock VALUES ('2023-01-08','GOOG',130.85,132.92,127.78,130.17),('2023-01-07','GOOG',120.32,134.07,115.83,133.11),('2023-01-06','GOOG',123.5,129.55,116.91,120.97),('2023-01-05','GOOG',107.72,127.05,104.5,123.37),('2023-01-04','GOOG',102.67,109.63,102.38,108.22),('2023-01-03','GOOG',90.16,107.51,89.77,104),('2023-01-02','GOOG',99.74,108.82,88.86,90.3),('2023-01-01','GOOG',89.83,101.58,85.57,99.87),('2022-01-12','GOOG',101.4,102.59,86.37,88.73),('2022-01-11','GOOG',95.59,101.45,83.45,101.45),('2022-01-10','GOOG',97.22,105.1,91.9,94.66),('2022-01-09','GOOG',109.2,112.64,96.03,96.15),('2022-01-08','GOOG',115.53,123.26,108.8,109.15),('2022-01-07','GOOG',108.34,120.43,104.76,116.64),('2022-01-06','GOOG',114.93,119.4,105.63,109.37),('2022-01-05','GOOG',113.91,123.14,102.21,114.04),('2022-01-04','GOOG',140.01,144.04,113.12,114.97),('2022-01-03','GOOG',134.48,144.16,125.86,139.65),('2022-01-02','GOOG',137.84,152.1,124.76,134.89),('2022-01-01','GOOG',144.48,146.61,124.64,135.7),('2021-01-12','GOOG',144.21,149.6,140.25,144.68),('2021-01-11','GOOG',148.16,151.85,142.07,142.45),('2021-01-10','GOOG',133.55,149.12,131.17,148.27),('2021-01-09','GOOG',145.65,146.82,133,133.27),('2021-01-08','GOOG',135.48,146.49,134.18,145.46),('2021-01-07','GOOG',124.85,140.01,124.85,135.22),('2021-01-06','GOOG',121.1,127.8,119.14,125.32),('2021-01-05','GOOG',120.14,122.15,111.5,120.58),('2021-01-04','GOOG',104.9,122.62,104.84,120.51),('2021-01-03','GOOG',102.83,106.44,100.5,103.43),('2021-01-02','GOOG',92.68,107.63,92.55,101.84),('2021-01-01','GOOG',87.88,96.74,84.95,91.79),('2020-01-12','GOOG',88.72,92.36,84.95,87.59),('2020-01-11','GOOG',81.41,90.9,80.8,88.04),('2020-01-10','GOOG',74.21,84.35,71.8,81.05),('2020-01-09','GOOG',81.83,86.66,70.33,73.48),('2020-01-08','GOOG',74.33,82.96,72.93,81.71),('2020-01-07','GOOG',70.56,79.35,70.49,74.15),('2020-01-06','GOOG',70.92,73.8,67.35,70.68),('2020-01-05','GOOG',66.43,72.05,64.95,71.45),('2020-01-04','GOOG',56.1,68,53.99,67.43),('2020-01-03','GOOG',67.58,70.51,50.68,58.14),('2020-01-02','GOOG',73.1,76.61,63.55,66.97),('2020-01-01','GOOG',67.08,75.16,67.08,71.71),('2023-01-08','MSFT',335.19,338.54,319.21,324.04),('2023-01-07','MSFT',339.19,366.78,327,335.92),('2023-01-06','MSFT',325.93,351.47,322.5,340.54),('2023-01-05','MSFT',306.97,335.94,303.4,328.39),('2023-01-04','MSFT',286.52,308.93,275.37,307.26),('2023-01-03','MSFT',250.76,289.27,245.61,288.3),('2023-01-02','MSFT',248,276.76,245.47,249.42),('2023-01-01','MSFT',243.08,249.83,219.35,247.81),('2022-01-12','MSFT',253.87,263.92,233.87,239.82),('2022-01-11','MSFT',234.6,255.33,213.43,255.14),('2022-01-10','MSFT',235.41,251.04,219.13,232.13),('2022-01-09','MSFT',258.87,267.45,232.73,232.9),('2022-01-08','MSFT',277.82,294.18,260.66,261.47),('2022-01-07','MSFT',256.39,282,245.94,280.74),('2022-01-06','MSFT',275.2,277.69,241.51,256.83),('2022-01-05','MSFT',277.71,290.88,246.44,271.87),('2022-01-04','MSFT',309.37,315.11,270,277.52),('2022-01-03','MSFT',296.4,315.95,270,308.31),('2022-01-02','MSFT',310.41,315.12,271.52,298.79),('2022-01-01','MSFT',335.35,338,276.05,310.98),('2021-01-12','MSFT',335.13,344.3,317.25,336.32),('2021-01-11','MSFT',331.36,349.67,326.37,330.59),('2021-01-10','MSFT',282.12,332,280.25,331.62),('2021-01-09','MSFT',302.87,305.32,281.62,281.92),('2021-01-08','MSFT',286.36,305.84,283.74,301.88),('2021-01-07','MSFT',269.61,290.15,269.6,284.91),('2021-01-06','MSFT',251.23,271.65,243,270.9),('2021-01-05','MSFT',253.4,254.35,238.07,249.68),('2021-01-04','MSFT',238.47,263.19,238.05,252.18),('2021-01-03','MSFT',235.9,241.05,224.26,235.77),('2021-01-02','MSFT',235.06,246.13,227.88,232.38),('2021-01-01','MSFT',222.53,242.64,211.94,231.96),('2020-01-12','MSFT',214.51,227.18,209.11,222.42),('2020-01-11','MSFT',204.29,228.12,200.12,214.07),('2020-01-10','MSFT',213.49,225.21,199.62,202.47),('2020-01-09','MSFT',225.51,232.86,196.25,210.33),('2020-01-08','MSFT',211.52,231.15,203.14,225.53),('2020-01-07','MSFT',203.14,216.38,197.51,205.01),('2020-01-06','MSFT',182.54,204.4,181.35,203.51),('2020-01-05','MSFT',175.8,187.51,173.8,183.25),('2020-01-04','MSFT',153,180.4,150.36,179.21),('2020-01-03','MSFT',165.31,175,132.52,157.71),('2020-01-02','MSFT',170.43,190.7,152,162.01),('2020-01-01','MSFT',158.78,174.05,156.51,170.23),('2023-01-08','AMZN',133.55,143.63,126.41,140.57),('2023-01-07','AMZN',130.82,136.65,125.92,133.68),('2023-01-06','AMZN',120.69,131.49,119.93,130.36),('2023-01-05','AMZN',104.95,122.92,101.15,120.58),('2023-01-04','AMZN',102.3,110.86,97.71,105.45),('2023-01-03','AMZN',93.87,103.49,88.12,103.29),('2023-01-02','AMZN',102.53,114,92.32,94.23),('2023-01-01','AMZN',85.46,103.49,81.43,103.13),('2022-01-12','AMZN',96.99,97.23,81.69,84),('2022-01-11','AMZN',103.99,104.58,85.87,96.54),('2022-01-10','AMZN',113.58,123,97.66,102.44),('2022-01-09','AMZN',126,136.49,112.06,113),('2022-01-08','AMZN',134.96,146.57,126.74,126.77),('2022-01-07','AMZN',106.29,137.65,105.85,134.95),('2022-01-06','AMZN',122.26,128.99,101.43,106.21),('2022-01-05','AMZN',122.4,126.22,101.26,120.21),('2022-01-04','AMZN',164.15,168.39,121.63,124.28),('2022-01-03','AMZN',152.73,170.83,133.57,163),('2022-01-02','AMZN',150,163.83,138.33,153.56),('2022-01-01','AMZN',167.55,171.4,135.35,149.57),('2021-01-12','AMZN',177.25,177.99,165.2,166.72),('2021-01-11','AMZN',168.09,188.11,164.18,175.35),('2021-01-10','AMZN',164.45,173.95,158.81,168.62),('2021-01-09','AMZN',174.82,177.5,163.7,164.25),('2021-01-08','AMZN',167.65,173.63,158.79,173.54),('2021-01-07','AMZN',171.73,188.65,165.35,166.38),('2021-01-06','AMZN',162.18,176.24,158.61,172.01),('2021-01-05','AMZN',174.24,174.33,156.37,161.15),('2021-01-04','AMZN',155.9,177.7,155.78,173.37),('2021-01-03','AMZN',156.39,159.1,144.05,154.7),('2021-01-02','AMZN',162.12,171.7,151.84,154.65),('2021-01-01','AMZN',163.5,168.19,154.3,160.31),('2020-01-12','AMZN',159.43,167.53,153.64,162.85),('2020-01-11','AMZN',153.09,168.34,147.51,158.4),('2020-01-10','AMZN',160.4,174.81,150.95,151.81),('2020-01-09','AMZN',174.48,177.61,143.55,157.44),('2020-01-08','AMZN',159.03,174.75,153.65,172.55),('2020-01-07','AMZN',137.9,167.21,137.7,158.23),('2020-01-06','AMZN',122.4,139.8,121.86,137.94),('2020-01-05','AMZN',116.84,126.27,112.82,122.12),('2020-01-04','AMZN',96.65,123.75,94.46,123.7),('2020-01-03','AMZN',95.32,99.82,81.3,97.49),('2020-01-02','AMZN',100.53,109.3,90.56,94.19),('2020-01-01','AMZN',93.75,102.79,90.77,100.44),('2023-01-08','AAPL',196.24,196.73,176.55,177.79),('2023-01-07','AAPL',193.78,198.23,186.6,196.45),('2023-01-06','AAPL',177.7,194.48,176.93,193.97),('2023-01-05','AAPL',169.28,179.35,164.31,177.25),('2023-01-04','AAPL',164.27,169.85,159.78,169.68),('2023-01-03','AAPL',146.83,165,143.9,164.9),('2023-01-02','AAPL',143.97,157.38,141.32,147.41),('2023-01-01','AAPL',130.28,147.23,124.17,144.29),('2022-01-12','AAPL',148.21,150.92,125.87,129.93),('2022-01-11','AAPL',155.08,155.45,134.38,148.03),('2022-01-10','AAPL',138.21,157.5,134.37,153.34),('2022-01-09','AAPL',156.64,164.26,138,138.2),('2022-01-08','AAPL',161.01,176.15,157.14,157.22),('2022-01-07','AAPL',136.04,163.63,135.66,162.51),('2022-01-06','AAPL',149.9,151.74,129.04,136.72),('2022-01-05','AAPL',156.71,166.48,132.61,148.84),('2022-01-04','AAPL',174.03,178.49,155.38,157.65),('2022-01-03','AAPL',164.7,179.61,150.1,174.61),('2022-01-02','AAPL',174.01,176.65,152,165.12),('2022-01-01','AAPL',177.83,182.94,154.7,174.78),('2021-01-12','AAPL',167.48,182.13,157.8,177.57),('2021-01-11','AAPL',148.99,165.7,147.48,165.3),('2021-01-10','AAPL',141.9,153.17,138.27,149.8),('2021-01-09','AAPL',152.83,157.26,141.27,141.5),('2021-01-08','AAPL',146.36,153.49,144.5,151.83),('2021-01-07','AAPL',136.6,150,135.76,145.86),('2021-01-06','AAPL',125.08,137.41,123.13,136.96),('2021-01-05','AAPL',132.04,134.07,122.25,124.61),('2021-01-04','AAPL',123.66,137.07,122.49,131.46),('2021-01-03','AAPL',123.75,128.72,116.21,122.15),('2021-01-02','AAPL',133.75,137.88,118.39,121.26),('2021-01-01','AAPL',133.52,145.09,126.38,131.96),('2020-01-12','AAPL',121.01,138.79,120.01,132.69),('2020-01-11','AAPL',109.11,121.99,107.32,119.05),('2020-01-10','AAPL',117.64,125.39,107.72,108.86),('2020-01-09','AAPL',132.76,137.98,103.1,115.81),('2020-01-08','AAPL',108.2,131,107.89,129.04),('2020-01-07','AAPL',91.28,106.42,89.14,106.26),('2020-01-06','AAPL',79.44,93.1,79.3,91.2),('2020-01-05','AAPL',71.56,81.06,71.46,79.49),('2020-01-04','AAPL',61.63,73.63,59.22,73.45),('2020-01-03','AAPL',70.57,76,53.15,63.57),('2020-01-02','AAPL',76.07,81.81,64.09,68.34),('2020-01-01','AAPL',74.06,81.96,73.19,77.38),('2023-01-08','NFLX',437.37,445.25,418.78,427.78),('2023-01-07','NFLX',439.76,485,411.88,438.97),('2023-01-06','NFLX',397.41,448.65,393.08,440.49),('2023-01-05','NFLX',329.44,405.11,315.62,395.23),('2023-01-04','NFLX',341.83,349.8,316.1,329.93),('2023-01-03','NFLX',321.55,345.84,285.33,345.48),('2023-01-02','NFLX',353.86,379.43,314.3,322.13),('2023-01-01','NFLX',298.06,369.02,288.7,353.86),('2022-01-12','NFLX',303.99,332.72,273.41,294.88),('2022-01-11','NFLX',296.17,312.71,252.09,305.53),('2022-01-10','NFLX',237.85,305.63,211.73,291.88),('2022-01-09','NFLX',220.72,250.49,214.69,235.44),('2022-01-08','NFLX',223.1,251.99,218.74,223.56),('2022-01-07','NFLX',176.49,230.75,169.7,224.9),('2022-01-06','NFLX',198.7,207.38,164.28,174.87),('2022-01-05','NFLX',191.71,204.38,162.71,197.44),('2022-01-04','NFLX',376.8,396.02,185.6,190.36),('2022-01-03','NFLX',391.6,396.5,329.82,374.59),('2022-01-02','NFLX',432.96,458.48,354.72,394.52),('2022-01-01','NFLX',605.61,609.99,351.46,427.14),('2021-01-12','NFLX',649.48,654.52,581.74,602.44),('2021-01-11','NFLX',689.06,700.99,640.01,641.9),('2021-01-10','NFLX',604.24,690.97,594.68,690.31),('2021-01-09','NFLX',569,619,568.08,610.34),('2021-01-08','NFLX',519,569.48,505.37,569.19),('2021-01-07','NFLX',525.72,557.54,504.66,517.57),('2021-01-06','NFLX',504.01,536.13,482.14,528.21),('2021-01-05','NFLX',512.65,518.95,478.54,502.81),('2021-01-04','NFLX',529.93,563.56,499,513.47),('2021-01-03','NFLX',545.57,556.99,492.85,521.66),('2021-01-02','NFLX',536.79,566.65,518.28,538.85),('2021-01-01','NFLX',539,593.29,485.67,532.39),('2020-01-12','NFLX',492.34,545.5,491.29,540.73),('2020-01-11','NFLX',478.87,518.73,463.41,490.7),('2020-01-10','NFLX',506.03,572.49,472.21,475.74),('2020-01-09','NFLX',532.6,557.39,458.6,500.03),('2020-01-08','NFLX',490.86,549.04,466.55,529.56),('2020-01-07','NFLX',454,575.37,454,488.88),('2020-01-06','NFLX',418.83,474.01,404.25,455.04),('2020-01-05','NFLX',415.1,458.97,397.86,419.73),('2020-01-04','NFLX',376.05,449.52,357.51,419.85),('2020-01-03','NFLX',373.11,393.52,290.25,375.5),('2020-01-02','NFLX',347.24,392.95,346.28,369.03),('2020-01-01','NFLX',326.1,359.85,321.2,345.09),('2023-01-08','META',317.34,324.12,298.29,302.69),('2023-01-07','META',286.65,326.11,284.85,318.6),('2023-01-06','META',265.9,289.79,258.88,286.98),('2023-01-05','META',238.62,268.65,229.85,264.72),('2023-01-04','META',208.84,241.69,207.13,240.32),('2023-01-03','META',174.59,212.17,171.43,211.94),('2023-01-02','META',148.03,197.16,147.06,174.94),('2023-01-01','META',122.82,153.19,122.28,148.97),('2022-01-12','META',119.2,124.67,112.46,120.34),('2022-01-11','META',94.33,118.74,88.09,118.1),('2022-01-10','META',137.14,142.39,92.6,93.16),('2022-01-09','META',163.58,171.39,134.12,135.68),('2022-01-08','META',157.25,183.1,155.23,162.93),('2022-01-07','META',160.31,183.85,154.85,159.1),('2022-01-06','META',196.51,202.03,154.25,161.25),('2022-01-05','META',201.16,224.3,176.11,193.64),('2022-01-04','META',224.55,236.86,169,200.47),('2022-01-03','META',209.87,231.15,185.82,222.36),('2022-01-02','META',314.55,328,190.22,211.03),('2022-01-01','META',338.3,343.09,289.01,313.26),('2021-01-12','META',330.29,352.71,299.5,336.35),('2021-01-11','META',326.04,353.83,323.2,324.46),('2021-01-10','META',341.61,345.02,308.11,323.57),('2021-01-09','META',379.59,384.33,338.15,339.39),('2021-01-08','META',358.1,382.76,347.7,379.38),('2021-01-07','META',346.82,377.55,334.5,356.3),('2021-01-06','META',330.15,358.14,323.48,347.71),('2021-01-05','META',326.17,333.78,298.19,328.73),('2021-01-04','META',298.4,331.81,296.04,325.08),('2021-01-03','META',260.82,299.7,253.5,294.53),('2021-01-02','META',259.52,276.6,254.04,257.62),('2021-01-01','META',274.78,286.79,244.61,258.33),('2020-01-12','META',279.16,291.78,264.63,273.16),('2020-01-11','META',264.6,297.38,257.34,276.97),('2020-01-10','META',265.35,285.24,254.82,263.11),('2020-01-09','META',294.71,303.6,244.13,261.9),('2020-01-08','META',252.65,304.67,247.43,293.2),('2020-01-07','META',228.5,255.85,226.9,253.67),('2020-01-06','META',224.59,245.19,207.11,227.07),('2020-01-05','META',201.6,240.9,198.76,225.09),('2020-01-04','META',161.62,209.69,150.83,204.71),('2020-01-03','META',194.03,197.24,137.1,166.8),('2020-01-02','META',203.44,218.76,181.82,192.47),('2020-01-01','META',206.75,224.2,201.06,201.91);

--
-- Table structure for table concerts
--

DROP TABLE IF EXISTS concerts;

CREATE TABLE concerts (
  artist_id int DEFAULT NULL,
  artist_name text,
  genre text,
  concert_revenue int DEFAULT NULL,
  year_of_formation int DEFAULT NULL,
  country text,
  number_of_members int DEFAULT NULL,
  album_released int DEFAULT NULL,
  label text
) ;


--
-- Add records for table concerts
--


INSERT INTO concerts VALUES (102,'The Rolling Stones','Rock',500000,1962,'United Kingdom',4,30,'Universal Music'),(103,'Taylor Swift','Pop',700000,2004,'United States',1,9,'Republic Records'),(104,'BTS','K-Pop',800000,2013,'South Korea',7,7,'Big Hit Music'),(105,'Adele','Pop',600000,2006,'United Kingdom',1,3,'Columbia Records'),(106,'Queen','Rock',550000,1970,'United Kingdom',4,16,'EMI'),(107,'Beyonce','R&B',750000,1997,'United States',1,8,'Parkwood Entertainment'),(108,'Ed Sheeran','Pop',650000,2005,'United Kingdom',1,6,'Atlantic Records'),(109,'Blackpink','K-Pop',450000,2016,'South Korea',4,5,'YG Entertainment'),(110,'Maroon 5','Pop',550000,1994,'United States',5,7,'Interscope Records'),(111,'Coldplay','Rock',600000,1996,'United Kingdom',4,9,'Parlophone'),(112,'Bruno Mars','Pop',700000,2010,'United States',1,3,'Atlantic Records'),(113,'EXO','K-Pop',480000,2011,'South Korea',9,6,'SM Entertainment'),(114,'Rihanna','R&B/Pop',550000,2005,'Barbados',1,8,'Def Jam Recordings'),(115,'Imagine Dragons','Rock',580000,2008,'United States',4,4,'Interscope Records'),(116,'Ariana Grande','Pop',720000,2008,'United States',1,6,'Republic Records'),(117,'Red Velvet','K-Pop',400000,2014,'South Korea',5,6,'SM Entertainment'),(118,'U2','Rock',650000,1976,'Ireland',4,14,'Island Records'),(119,'Justin Bieber','Pop',680000,2008,'Canada',1,8,'Def Jam Recordings'),(120,'Seventeen','K-Pop',430000,2015,'South Korea',13,3,'Pledis Entertainment'),(121,'The Weeknd','R&B/Pop',620000,2010,'Canada',1,5,'Republic Records'),(122,'The Beatles','Rock',520000,1960,'United Kingdom',4,13,'Apple Records'),(123,'Billie Eilish','Pop',580000,2015,'United States',1,2,'Interscope Records'),(124,'NCT 127','K-Pop',410000,2016,'South Korea',9,4,'SM Entertainment'),(125,'Pink Floyd','Rock',670000,1965,'United Kingdom',4,15,'EMI'),(126,'Demi Lovato','Pop',520000,2008,'United States',1,7,'Island Records'),(127,'SHINee','K-Pop',470000,2008,'South Korea',4,7,'SM Entertainment'),(128,'Guns N\' Roses','Rock',590000,1985,'United States',5,6,'Geffen Records'),(129,'Selena Gomez','Pop',530000,2009,'United States',1,5,'Interscope Records'),(130,'Stray Kids','K-Pop',390000,2017,'South Korea',8,3,'JYP Entertainment'),(131,'Bon Jovi','Rock',560000,1983,'United States',5,15,'Island Records'),(132,'Camila Cabello','Pop',500000,2012,'United States',1,2,'Epic Records'),(133,'SuperM','K-Pop',450000,2019,'South Korea',7,1,'SM Entertainment'),(134,'Linkin Park','Rock',590000,1996,'United States',6,7,'Warner Records'),(135,'ITZY','K-Pop',380000,2019,'South Korea',5,2,'JYP Entertainment'),(136,'Panic! At The Disco','Rock',530000,2004,'United States',1,6,'Fueled by Ramen'),(137,'ABBA','Pop',510000,1972,'Sweden',4,8,'Polar Music'),(138,'Eminem','Hip-Hop',620000,1996,'United States',1,11,'Aftermath Records'),(139,'One Direction','Pop',580000,2010,'United Kingdom',5,5,'Syco Music'),(140,'Black Sabbath','Rock',540000,1968,'United Kingdom',4,19,'Vertigo Records'),(141,'Sia','Pop',510000,2000,'Australia',1,9,'RCA Records'),(142,'Got7','K-Pop',470000,2014,'South Korea',7,13,'JYP Entertainment'),(143,'Red Hot Chili Peppers','Rock',660000,1983,'United States',4,11,'Warner Records'),(144,'Halsey','Pop',520000,2014,'United States',1,4,'Capitol Records'),(145,'Super Junior','K-Pop',490000,2005,'South Korea',10,10,'SM Entertainment'),(146,'Fleetwood Mac','Rock',600000,1967,'United Kingdom',5,17,'Reprise Records'),(147,'Zayn','Pop',480000,2010,'United Kingdom',1,3,'RCA Records'),(148,'The Who','Rock',540000,1964,'United Kingdom',4,11,'Polydor Records'),(149,'Lana Del Rey','Pop',530000,2005,'United States',1,8,'Interscope Records'),(150,'Monsta X','K-Pop',470000,2015,'South Korea',7,9,'Starship Entertainment'),(151,'The Eagles','Rock',610000,1971,'United States',5,7,'Asylum Records');


SET SQL_SAFE_UPDATES =0 ;
SET GLOBAL log_bin_trust_function_creators = 1; 