-- Create database if not exists Movies ;

Use excelrdb;
drop table if exists Role;

drop table if exists actor;
CREATE TABLE Actor(
	ActorID int NOT NULL PRIMARY KEY auto_increment,
	FirstName nvarchar(100) NOT NULL,
	FamilyName nvarchar(100) NULL,
	FullName  nvarchar(1000),
	DoB datetime NULL,
	DoD datetime NULL,
	Gender nvarchar(20) NULL);
    
drop table if exists film;

drop table if exists certificate;
-- table of certificates
CREATE TABLE Certificate(
	CertificateID int  NOT NULL PRIMARY KEY auto_increment,
	Certificate nvarchar(10) NOT NULL);

drop table if exists country;
-- table of countries
CREATE TABLE Country(
	CountryID int NOT NULL PRIMARY KEY auto_increment,
	Country nvarchar(255) NOT NULL);

drop table if exists Director;
-- table of directors
CREATE TABLE Director(
	DirectorID int NOT NULL PRIMARY KEY auto_increment,
	FirstName nvarchar(100) NOT NULL,
	FamilyName nvarchar(100) NULL,
	FullName  nvarchar(1000),
	DoB datetime NULL,
	DoD datetime NULL,
	Gender nvarchar(20) NULL);


-- table of films
CREATE TABLE Film(
	FilmID int NOT NULL PRIMARY KEY auto_increment,
	Title nvarchar(255) NOT NULL,
	ReleaseDate datetime NULL,
	DirectorID int NULL,
	StudioID int NULL,
	Review nvarchar(1000) NULL,
	CountryID int NULL,
	LanguageID int NULL,
	GenreID int NULL,
	RunTimeMinutes smallint NULL,
	CertificateID int NULL,
	BudgetDollars bigint NULL,
	BoxOfficeDollars bigint NULL,
	OscarNominations tinyint NULL,
	OscarWins tinyint NULL);

drop table if exists Genre;
-- table of genres
 CREATE TABLE Genre(
	GenreID int NOT NULL PRIMARY KEY auto_increment,
	Genre nvarchar(50) NOT NULL);

drop table if exists Language;
-- table of languages
CREATE TABLE Language(
	LanguageID int  NOT NULL PRIMARY KEY auto_increment,
	Language nvarchar(255) NOT NULL);


-- table of roles
CREATE TABLE Role(
	RoleID int NOT NULL PRIMARY KEY auto_increment,
	Role nvarchar(255) NULL,
	FilmID int NULL,
	ActorID int NULL);


drop table if exists studio;
-- table of studios
CREATE TABLE Studio(
	StudioID int NOT NULL PRIMARY KEY auto_increment,
	Studio nvarchar(255) NOT NULL);


-- use Movies;

ALTER TABLE Film ADD  CONSTRAINT FK_Film_Certificate FOREIGN KEY(CertificateID)
REFERENCES Certificate (CertificateID);

ALTER TABLE Film    ADD  CONSTRAINT FK_Film_Country FOREIGN KEY(CountryID)
REFERENCES Country (CountryID);

ALTER TABLE Film    ADD  CONSTRAINT FK_Film_Director FOREIGN KEY(DirectorID)
REFERENCES Director (DirectorID);

ALTER TABLE Film    ADD  CONSTRAINT FK_Film_Genre FOREIGN KEY(GenreID)
REFERENCES Genre (GenreID);

ALTER TABLE Film    ADD  CONSTRAINT FK_Film_Language FOREIGN KEY(LanguageID)
REFERENCES Language (LanguageID);

ALTER TABLE Film    ADD  CONSTRAINT FK_Film_Studio FOREIGN KEY(StudioID)
REFERENCES Studio (StudioID);

ALTER TABLE Role    ADD  CONSTRAINT FK_Role_Actor FOREIGN KEY(ActorID)
REFERENCES Actor (ActorID);

ALTER TABLE Role    ADD  CONSTRAINT FK_Role_Film FOREIGN KEY(FilmID)
REFERENCES Film (FilmID);

INSERT Certificate ( Certificate) VALUES (N'U');
INSERT Certificate ( Certificate) VALUES (N'PG');
INSERT Certificate ( Certificate) VALUES (N'12');
INSERT Certificate ( Certificate) VALUES (N'12A');
INSERT Certificate ( Certificate) VALUES (N'15');
INSERT Certificate ( Certificate) VALUES (N'18');
INSERT Certificate ( Certificate) VALUES (N'Unknown');

INSERT Country (CountryID, Country) VALUES (48, N'China');
INSERT Country (CountryID, Country) VALUES (79, N'France');
INSERT Country (CountryID, Country) VALUES (118, N'Japan');
INSERT Country (CountryID, Country) VALUES (167, N'New Zealand');
INSERT Country (CountryID, Country) VALUES (240, N'United Kingdom');
INSERT Country (CountryID, Country) VALUES (241, N'United States');
INSERT Country (CountryID, Country) VALUES (257, N'Germany');
INSERT Country (CountryID, Country) VALUES (258, N'Russia');
INSERT Country (CountryID, Country) VALUES (259, N'Taiwan');
INSERT Country (CountryID, Country) VALUES (261, N'Australia');
INSERT Country (CountryID, Country) VALUES (262, N'Canada');
INSERT Country (CountryID, Country) VALUES (263, N'Brazil');
INSERT Country (CountryID, Country) VALUES (264, N'Italy');
INSERT Country (CountryID, Country) VALUES (265, N'Hong Kong');
INSERT Country (CountryID, Country) VALUES (266, N'Indonesia');
INSERT Country (CountryID, Country) VALUES (267, N'Thailand');
INSERT Country (CountryID, Country) VALUES (268, N'South Africa');
INSERT Country (CountryID, Country) VALUES (269, N'Netherlands');
INSERT Country (CountryID, Country) VALUES (270, N'Denmark');
INSERT Country (CountryID, Country) VALUES (271, N'Sweden');
INSERT Country (CountryID, Country) VALUES (455, N'Afghanistan');
INSERT Country (CountryID, Country) VALUES (456, N'Albania');
INSERT Country (CountryID, Country) VALUES (457, N'Algeria');
INSERT Country (CountryID, Country) VALUES (458, N'Andorra');
INSERT Country (CountryID, Country) VALUES (459, N'Anla');
INSERT Country (CountryID, Country) VALUES (460, N'Antigua and Barbuda');
INSERT Country (CountryID, Country) VALUES (461, N'Argentina');
INSERT Country (CountryID, Country) VALUES (462, N'Armenia');
INSERT Country (CountryID, Country) VALUES (463, N'Aruba');
INSERT Country (CountryID, Country) VALUES (464, N'Austria');
INSERT Country (CountryID, Country) VALUES (465, N'Azerbaijan');
INSERT Country (CountryID, Country) VALUES (466, N'Bahrain');
INSERT Country (CountryID, Country) VALUES (467, N'Bangladesh');
INSERT Country (CountryID, Country) VALUES (468, N'Barbados');
INSERT Country (CountryID, Country) VALUES (469, N'Belarus');
INSERT Country (CountryID, Country) VALUES (470, N'Belgium');
INSERT Country (CountryID, Country) VALUES (471, N'Belize');
INSERT Country (CountryID, Country) VALUES (472, N'Benin');
INSERT Country (CountryID, Country) VALUES (473, N'Bhutan');
INSERT Country (CountryID, Country) VALUES (474, N'Bolivia');
INSERT Country (CountryID, Country) VALUES (475, N'Bosnia and Herzevina');
INSERT Country (CountryID, Country) VALUES (476, N'Botswana');
INSERT Country (CountryID, Country) VALUES (477, N'Brunei');
INSERT Country (CountryID, Country) VALUES (478, N'Bulgaria');
INSERT Country (CountryID, Country) VALUES (479, N'Burkina Faso');
INSERT Country (CountryID, Country) VALUES (480, N'Burma');
INSERT Country (CountryID, Country) VALUES (481, N'Burundi');
INSERT Country (CountryID, Country) VALUES (482, N'Cabo Verde');
INSERT Country (CountryID, Country) VALUES (483, N'Cambodia');
INSERT Country (CountryID, Country) VALUES (484, N'Cameroon');
INSERT Country (CountryID, Country) VALUES (485, N'Central African Republic');
INSERT Country (CountryID, Country) VALUES (486, N'Chad');
INSERT Country (CountryID, Country) VALUES (487, N'Chile');
INSERT Country (CountryID, Country) VALUES (488, N'Colombia');
INSERT Country (CountryID, Country) VALUES (489, N'Comoros');
INSERT Country (CountryID, Country) VALUES (490, N'Costa Rica');
INSERT Country (CountryID, Country) VALUES (491, N'Cote d''Ivoire');
INSERT Country (CountryID, Country) VALUES (492, N'Croatia');
INSERT Country (CountryID, Country) VALUES (493, N'Cuba');
INSERT Country (CountryID, Country) VALUES (494, N'Curacao');
INSERT Country (CountryID, Country) VALUES (495, N'Cyprus');
INSERT Country (CountryID, Country) VALUES (496, N'Czechia');
INSERT Country (CountryID, Country) VALUES (497, N'Democratic Republic of the Con');
INSERT Country (CountryID, Country) VALUES (498, N'Djibouti');
INSERT Country (CountryID, Country) VALUES (499, N'Dominica');
INSERT Country (CountryID, Country) VALUES (500, N'Dominican Republic');
INSERT Country (CountryID, Country) VALUES (501, N'East Timor');
INSERT Country (CountryID, Country) VALUES (502, N'Ecuador');
INSERT Country (CountryID, Country) VALUES (503, N'Egypt');
INSERT Country (CountryID, Country) VALUES (504, N'El Salvador');
INSERT Country (CountryID, Country) VALUES (505, N'Equatorial Guinea');
INSERT Country (CountryID, Country) VALUES (506, N'Eritrea');
INSERT Country (CountryID, Country) VALUES (507, N'Estonia');
INSERT Country (CountryID, Country) VALUES (508, N'Ethiopia');
INSERT Country (CountryID, Country) VALUES (509, N'Fiji');
INSERT Country (CountryID, Country) VALUES (510, N'Finland');
INSERT Country (CountryID, Country) VALUES (511, N'Gabon');
INSERT Country (CountryID, Country) VALUES (512, N'Georgia');
INSERT Country (CountryID, Country) VALUES (513, N'Ghana');
INSERT Country (CountryID, Country) VALUES (514, N'Greece');
INSERT Country (CountryID, Country) VALUES (515, N'Grenada');
INSERT Country (CountryID, Country) VALUES (516, N'Guatemala');
INSERT Country (CountryID, Country) VALUES (517, N'Guinea');
INSERT Country (CountryID, Country) VALUES (518, N'Guinea-Bissau');
INSERT Country (CountryID, Country) VALUES (519, N'Guyana');
INSERT Country (CountryID, Country) VALUES (520, N'Haiti');
INSERT Country (CountryID, Country) VALUES (521, N'Holy See');
INSERT Country (CountryID, Country) VALUES (522, N'Honduras');
INSERT Country (CountryID, Country) VALUES (523, N'Hungary');
INSERT Country (CountryID, Country) VALUES (524, N'Iceland');
INSERT Country (CountryID, Country) VALUES (525, N'India');
INSERT Country (CountryID, Country) VALUES (526, N'Iran');
INSERT Country (CountryID, Country) VALUES (527, N'Iraq');
INSERT Country (CountryID, Country) VALUES (528, N'Ireland');
INSERT Country (CountryID, Country) VALUES (529, N'Israel');
INSERT Country (CountryID, Country) VALUES (530, N'Jamaica');
INSERT Country (CountryID, Country) VALUES (531, N'Jordan');
INSERT Country (CountryID, Country) VALUES (532, N'Kazakhstan');
INSERT Country (CountryID, Country) VALUES (533, N'Kenya');

INSERT Country (CountryID, Country) VALUES (534, N'Kiribati');
INSERT Country (CountryID, Country) VALUES (535, N'Kosovo');
INSERT Country (CountryID, Country) VALUES (536, N'Kuwait');
INSERT Country (CountryID, Country) VALUES (537, N'Kyrgyzstan');
INSERT Country (CountryID, Country) VALUES (538, N'Laos');
INSERT Country (CountryID, Country) VALUES (539, N'Latvia');
INSERT Country (CountryID, Country) VALUES (540, N'Lebanon');
INSERT Country (CountryID, Country) VALUES (541, N'Lesotho');
INSERT Country (CountryID, Country) VALUES (542, N'Liberia');
INSERT Country (CountryID, Country) VALUES (543, N'Libya');
INSERT Country (CountryID, Country) VALUES (544, N'Liechtenstein');
INSERT Country (CountryID, Country) VALUES (545, N'Lithuania');
INSERT Country (CountryID, Country) VALUES (546, N'Luxembourg');
INSERT Country (CountryID, Country) VALUES (547, N'Macau');
INSERT Country (CountryID, Country) VALUES (548, N'Macedonia');
INSERT Country (CountryID, Country) VALUES (549, N'Madagascar');
INSERT Country (CountryID, Country) VALUES (550, N'Malawi');
INSERT Country (CountryID, Country) VALUES (551, N'Malaysia');
INSERT Country (CountryID, Country) VALUES (552, N'Maldives');
INSERT Country (CountryID, Country) VALUES (553, N'Mali');
INSERT Country (CountryID, Country) VALUES (554, N'Malta');
INSERT Country (CountryID, Country) VALUES (555, N'Marshall Islands');
INSERT Country (CountryID, Country) VALUES (556, N'Mauritania');
INSERT Country (CountryID, Country) VALUES (557, N'Mauritius');
INSERT Country (CountryID, Country) VALUES (558, N'Mexico');
INSERT Country (CountryID, Country) VALUES (559, N'Micronesia');
INSERT Country (CountryID, Country) VALUES (560, N'Moldova');
INSERT Country (CountryID, Country) VALUES (561, N'Monaco');
INSERT Country (CountryID, Country) VALUES (562, N'Monlia');
INSERT Country (CountryID, Country) VALUES (563, N'Montenegro');
INSERT Country (CountryID, Country) VALUES (564, N'Morocco');
INSERT Country (CountryID, Country) VALUES (565, N'Mozambique');
INSERT Country (CountryID, Country) VALUES (566, N'Namibia');
INSERT Country (CountryID, Country) VALUES (567, N'Nauru');
INSERT Country (CountryID, Country) VALUES (568, N'Nepal');
INSERT Country (CountryID, Country) VALUES (569, N'Nicaragua');
INSERT Country (CountryID, Country) VALUES (570, N'Niger');
INSERT Country (CountryID, Country) VALUES (571, N'Nigeria');
INSERT Country (CountryID, Country) VALUES (572, N'North Korea');
INSERT Country (CountryID, Country) VALUES (573, N'Norway');
INSERT Country (CountryID, Country) VALUES (574, N'Oman');
INSERT Country (CountryID, Country) VALUES (575, N'Pakistan');
INSERT Country (CountryID, Country) VALUES (576, N'Palau');
INSERT Country (CountryID, Country) VALUES (577, N'Palestinian Territories');
INSERT Country (CountryID, Country) VALUES (578, N'Panama');
INSERT Country (CountryID, Country) VALUES (579, N'Papua New Guinea');
INSERT Country (CountryID, Country) VALUES (580, N'Paraguay');
INSERT Country (CountryID, Country) VALUES (581, N'Peru');
INSERT Country (CountryID, Country) VALUES (582, N'Philippines');
INSERT Country (CountryID, Country) VALUES (583, N'Poland');
INSERT Country (CountryID, Country) VALUES (584, N'Portugal');
INSERT Country (CountryID, Country) VALUES (585, N'Qatar');
INSERT Country (CountryID, Country) VALUES (586, N'Republic of the Con');
INSERT Country (CountryID, Country) VALUES (587, N'Romania');
INSERT Country (CountryID, Country) VALUES (588, N'Rwanda');
INSERT Country (CountryID, Country) VALUES (589, N'Saint Kitts and Nevis');
INSERT Country (CountryID, Country) VALUES (590, N'Saint Lucia');
INSERT Country (CountryID, Country) VALUES (591, N'Saint Vincent and the Grenadines');
INSERT Country (CountryID, Country) VALUES (592, N'Samoa');
INSERT Country (CountryID, Country) VALUES (593, N'San Marino');
INSERT Country (CountryID, Country) VALUES (594, N'Sao Tome and Principe');
INSERT Country (CountryID, Country) VALUES (595, N'Saudi Arabia');
INSERT Country (CountryID, Country) VALUES (596, N'Senegal');
INSERT Country (CountryID, Country) VALUES (597, N'Serbia');
INSERT Country (CountryID, Country) VALUES (598, N'Seychelles');
INSERT Country (CountryID, Country) VALUES (599, N'Sierra Leone');
INSERT Country (CountryID, Country) VALUES (600, N'Singapore');
INSERT Country (CountryID, Country) VALUES (601, N'Sint Maarten');
INSERT Country (CountryID, Country) VALUES (602, N'Slovakia');
INSERT Country (CountryID, Country) VALUES (603, N'Slovenia');
INSERT Country (CountryID, Country) VALUES (604, N'Solomon Islands');
INSERT Country (CountryID, Country) VALUES (605, N'Somalia');
INSERT Country (CountryID, Country) VALUES (606, N'South Korea');
INSERT Country (CountryID, Country) VALUES (607, N'South Sudan');
INSERT Country (CountryID, Country) VALUES (608, N'Spain');
INSERT Country (CountryID, Country) VALUES (609, N'Sri Lanka');
INSERT Country (CountryID, Country) VALUES (610, N'Sudan');
INSERT Country (CountryID, Country) VALUES (611, N'Suriname');
INSERT Country (CountryID, Country) VALUES (612, N'Swaziland');
INSERT Country (CountryID, Country) VALUES (613, N'Switzerland');
INSERT Country (CountryID, Country) VALUES (614, N'Syria');
INSERT Country (CountryID, Country) VALUES (615, N'Tajikistan');
INSERT Country (CountryID, Country) VALUES (616, N'Tanzania');
INSERT Country (CountryID, Country) VALUES (617, N'The Bahamas');
INSERT Country (CountryID, Country) VALUES (618, N'The Gambia');
INSERT Country (CountryID, Country) VALUES (619, N'Timor-Leste');
INSERT Country (CountryID, Country) VALUES (620, N'To');
INSERT Country (CountryID, Country) VALUES (621, N'Tonga');
INSERT Country (CountryID, Country) VALUES (622, N'Trinidad and Toba');
INSERT Country (CountryID, Country) VALUES (623, N'Tunisia');
INSERT Country (CountryID, Country) VALUES (624, N'Turkey');
INSERT Country (CountryID, Country) VALUES (625, N'Turkmenistan');
INSERT Country (CountryID, Country) VALUES (626, N'Tuvalu');
INSERT Country (CountryID, Country) VALUES (627, N'Uganda');
INSERT Country (CountryID, Country) VALUES (628, N'Ukraine');
INSERT Country (CountryID, Country) VALUES (629, N'United Arab Emirates');
INSERT Country (CountryID, Country) VALUES (630, N'Uruguay');
INSERT Country (CountryID, Country) VALUES (631, N'Uzbekistan');
INSERT Country (CountryID, Country) VALUES (632, N'Vanuatu');
INSERT Country (CountryID, Country) VALUES (633, N'Venezuela');
INSERT Country (CountryID, Country) VALUES (634, N'Vietnam');
INSERT Country (CountryID, Country) VALUES (635, N'Yemen');
INSERT Country (CountryID, Country) VALUES (636, N'Zambia');
INSERT Country (CountryID, Country) VALUES (637, N'Zimbabwe');

INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (4, N'Steven', N'Spielberg', CAST(N'1946-12-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (5, N'Joel', N'Coen', CAST(N'1954-11-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (6, N'Ethan', N'Coen', CAST(N'1957-09-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (7, N'George', N'Lucas', CAST(N'1944-05-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (8, N'Ang', N'Lee', CAST(N'1954-10-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (9, N'Martin', N'Scorsese', CAST(N'1942-11-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (10, N'Clint', N'Eastwood', CAST(N'1930-05-31T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (11, N'Sam', N'Raimi', CAST(N'1959-10-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (12, N'Peter', N'Jackson', CAST(N'1961-10-31T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (14, N'Bryan', N'Singer', CAST(N'1965-09-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (15, N'James', N'Cameron', CAST(N'1954-08-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (16, N'Tom', N'Shadyac', CAST(N'1958-12-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (17, N'Kevin', N'Reynolds', CAST(N'1952-01-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (18, N'Michael', N'Bay', CAST(N'1965-02-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (19, N'David', N'Yates', CAST(N'1963-11-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (20, N'Robert', N'Zemeckis', CAST(N'1952-05-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (21, N'Steve', N'Hickner', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (22, N'Gore', N'Verbinski', CAST(N'1964-03-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (23, N'Francis', N'Lawrence', CAST(N'1970-03-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (24, N'Brad', N'Bird', CAST(N'1957-09-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (25, N'Wolfgang', N'Peterson', CAST(N'1941-03-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (26, N'Mike', N'Newell', CAST(N'1942-03-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (27, N'Christopher', N'Nolan', CAST(N'1970-07-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (28, N'Tim', N'Burton', CAST(N'1958-08-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (29, N'Lee', N'Tamahori', CAST(N'1950-06-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (30, N'Richard', N'Donner', CAST(N'1930-04-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (31, N'Barry', N'Sonnenfeld', CAST(N'1953-04-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (32, N'Rob', N'Cohen', CAST(N'1949-03-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (33, N'Hironobu', N'Sakaguchi', CAST(N'1962-11-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (34, N'Michael', N'Mann', CAST(N'1943-02-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (35, N'Michael', N'Apted', CAST(N'1941-02-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (36, N'Peter', N'Weir', CAST(N'1944-08-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (37, N'Paul', N'Greengrass', CAST(N'1955-08-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (38, N'Alfonso', N'Cuaron', CAST(N'1961-11-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (39, N'Lilly (Andy)', N'Wachowski', CAST(N'1967-12-29T00:00:00.000' AS DateTime), NULL, N'Transgender');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (40, N'Ron', N'Howard', CAST(N'1954-03-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (41, N'Chris', N'Columbus', CAST(N'1958-09-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (42, N'Roland', N'Emmerich', CAST(N'1955-11-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (43, N'Edward', N'Zwick', CAST(N'1952-10-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (44, N'J. J.', N'Abrams', CAST(N'1966-06-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (45, N'John', N'Woo', CAST(N'1946-09-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (46, N'Pete', N'Docter', CAST(N'1968-08-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (47, N'Roger', N'Spottiswoode', CAST(N'1945-01-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (48, N'Steven', N'Soderbergh', CAST(N'1963-01-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (49, N'Len', N'Wiseman', CAST(N'1973-03-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (50, N'Frank', N'Coraci', CAST(N'1966-02-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (51, N'Ridley', N'Scott', CAST(N'1937-11-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (52, N'Doug', N'Liman', CAST(N'1965-07-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (53, N'Alex', N'Proyas', CAST(N'1963-09-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (54, N'Martin', N'Campbell', CAST(N'1940-10-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (55, N'Jean-Christophe', N'Comar', CAST(N'1957-07-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (56, N'Luc', N'Besson', CAST(N'1959-03-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (57, N'Brett', N'Ratner', CAST(N'1969-03-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (58, N'John', N'McTiernan', CAST(N'1951-01-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (59, N'Brian', N'De Palma', CAST(N'1940-09-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (60, N'Mimi', N'Leder', CAST(N'1952-01-26T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (61, N'Sam', N'Mendes', CAST(N'1965-08-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (62, N'M. Night', N'Shyamalan', CAST(N'1970-08-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (63, N'Adam', N'McKay', CAST(N'1968-04-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (64, N'David', N'Silverman', CAST(N'1957-03-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (65, N'Mel', N'Gibson', CAST(N'1956-01-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (66, N'Chris', N'Miller', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (67, N'Andrew', N'Adamson', CAST(N'1966-12-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (68, N'Jonathan', N'Frakes', CAST(N'1952-08-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (69, N'Renny', N'Harlin', CAST(N'1959-03-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (70, N'David', N'Fincher', CAST(N'1962-08-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (71, N'Guillermo', N'del Toro', CAST(N'1964-10-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (72, N'Paul', N'Verhoeven', CAST(N'1938-07-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (73, N'Phillip', N'Noyce', CAST(N'1950-04-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (74, N'Paul', N'Anderson', CAST(N'1965-03-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (75, N'Todd', N'Phillips', CAST(N'1970-12-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (76, N'Zack', N'Snyder', CAST(N'1966-03-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (77, N'Stuart', N'Baird', CAST(N'1947-11-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (78, N'Tony', N'Scott', CAST(N'1944-06-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (79, N'Quentin', N'Tarantino', CAST(N'1963-03-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (80, N'Tom', N'Dey', CAST(N'1965-04-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (81, N'David', N'Dobkin', CAST(N'1969-06-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (82, N'Stephen', N'Norrington', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (83, N'Andrew', N'Davis', CAST(N'1946-11-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (84, N'John', N'Glen', CAST(N'1932-05-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (85, N'Frank', N'Miller', CAST(N'1957-01-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (86, N'Joss', N'Whedon', CAST(N'1964-06-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (87, N'David', N'Carson', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (88, N'Irvin', N'Kershner', CAST(N'1923-04-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (89, N'Robert', N'Wise', CAST(N'1914-09-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (90, N'Richard', N'Marquand', CAST(N'1938-04-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (91, N'David', N'Cronenberg', CAST(N'1943-03-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (92, N'Lewis', N'Gilbert', CAST(N'1920-03-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (93, N'Andrew', N'Dominik', CAST(N'1967-10-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (94, N'Ivan', N'Reitman', CAST(N'1946-10-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (95, N'William', N'Shatner', CAST(N'1931-03-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (96, N'Nicholas', N'Meyer', CAST(N'1945-12-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (97, N'Paul Michael', N'Glaser', CAST(N'1943-03-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (98, N'Kevin', N'Costner', CAST(N'1955-01-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (99, N'Leonard', N'Nimoy', CAST(N'1931-03-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (100, N'John', N'Guillermin', CAST(N'1925-11-11T00:00:00.000' AS DateTime), CAST(N'2015-09-27T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (101, N'Richard', N'Attenborough', CAST(N'1923-08-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (102, N'Kurt', N'Wimmer', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (103, N'Robert', N'Rodriguez', CAST(N'1968-06-20T00:00:00.000' AS DateTime), NULL, N'Male');

INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (104, N'Larry', N'Charles', CAST(N'1956-02-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (105, N'Zhang', N'Yimou', CAST(N'1951-11-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (106, N'Edgar', N'Wright', CAST(N'1974-04-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (108, N'Danny', N'Boyle', CAST(N'1956-10-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (109, N'Oliver', N'Hirschbiegel', CAST(N'1957-12-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (111, N'Jean-Pierre', N'Jeunet', CAST(N'1953-09-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (112, N'Terence', N'Young', CAST(N'1915-06-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (113, N'Peter R.', N'Hunt', CAST(N'1925-03-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (114, N'Stanley', N'Tong', CAST(N'1960-04-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (115, N'Guy', N'Hamilton', CAST(N'1922-09-16T00:00:00.000' AS DateTime), CAST(N'2016-04-20T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (116, N'George', N'Clooney', CAST(N'1961-05-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (117, N'Michael', N'Anderson', CAST(N'1920-01-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (118, N'Sammo', N'Hung', CAST(N'1952-01-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (119, N'Timur', N'Bekmambetov', CAST(N'1961-06-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (120, N'Florian Henckel', N'von Donnersmarck', CAST(N'1973-05-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (121, N'Merian C.', N'Cooper', CAST(N'1893-10-24T00:00:00.000' AS DateTime), CAST(N'1973-04-21T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (122, N'Akira', N'Kurosawa', CAST(N'1910-10-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (123, N'Morgan', N'Spurlock', CAST(N'1970-11-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (124, N'Ken', N'Hughes', CAST(N'1922-01-19T00:00:00.000' AS DateTime), CAST(N'2001-04-28T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (125, N'David', N'Bowers', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (126, N'Sam', N'Wood', CAST(N'1883-07-10T00:00:00.000' AS DateTime), CAST(N'1949-09-22T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (127, N'Cecil B.', N'DeMille', CAST(N'1881-08-12T00:00:00.000' AS DateTime), CAST(N'1959-01-21T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (128, N'James', N'Gunn', CAST(N'1970-08-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (130, N'Robert', N'Stromberg', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (132, N'Anthony', N'Russo', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (135, N'Marc', N'Webb', CAST(N'1974-08-31T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (137, N'Matt', N'Reeves', CAST(N'1966-04-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (138, N'Allan', N'Dwan', CAST(N'1885-04-03T00:00:00.000' AS DateTime), CAST(N'1981-12-28T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (139, N'Chris', N'Buck', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (140, N'Shane', N'Black', CAST(N'1961-12-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (141, N'Pierre', N'Coffin', CAST(N'1967-03-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (142, N'Justin', N'Lin', CAST(N'1971-10-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (143, N'Dan', N'Scanlon', CAST(N'1976-06-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (144, N'Alan', N'Taylor', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (145, N'Lana (Larry)', N'Wachowski', CAST(N'1965-06-21T00:00:00.000' AS DateTime), NULL, N'Transgender');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (146, N'Christopher', N'McQuarrie', CAST(N'1968-06-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (147, N'James', N'Wan', CAST(N'1977-02-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (148, N'Colin', N'Trevorrow', CAST(N'1976-09-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (149, N'Mike', N'Thurmeier', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (150, N'Bill', N'Condon', CAST(N'1955-10-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (151, N'Eric', N'Darnell', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (152, N'Gary', N'Ross', CAST(N'1956-11-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (153, N'Rob', N'Marshall', CAST(N'1960-10-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (154, N'Jennifer', N'Yuh Nelson', CAST(N'1972-05-07T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (155, N'Raja', N'Gosnell', CAST(N'1958-12-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (156, N'John', N'Lasseter', CAST(N'1957-01-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (157, N'Lee', N'Unkrich', CAST(N'1967-08-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (158, N'Mike', N'Mitchell', CAST(N'1970-10-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (159, N'David', N'Slade', CAST(N'1969-09-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (160, N'Jon', N'Favreau', CAST(N'1966-10-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (161, N'Nathan', N'Greno', CAST(N'1975-03-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (162, N'Dean', N'DeBlois', CAST(N'1970-06-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (163, N'Carlos', N'Saldanha', CAST(N'1965-01-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (164, N'Chris', N'Weitz', CAST(N'1969-11-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (165, N'Guy', N'Ritchie', CAST(N'1968-09-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (166, N'Mark', N'Osborne', CAST(N'1970-09-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (167, N'Peter', N'Berg', CAST(N'1962-03-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (168, N'Phyllida', N'Lloyd', CAST(N'1957-06-17T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (169, N'Marc', N'Forster', CAST(N'1969-11-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (170, N'Andrew', N'Stanton', CAST(N'1965-12-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (171, N'Jon', N'Turteltaub', CAST(N'1963-08-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (172, N'Shawn', N'Levy', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (173, N'George', N'Miller', CAST(N'1945-03-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (174, N'Andy', N'Tennant', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (175, N'Jay', N'Roach', CAST(N'1957-06-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (176, N'Vicky', N'Jenson', NULL, NULL, N'Female');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (177, N'Jonathan', N'Mostow', CAST(N'1961-11-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (178, N'Byron', N'Howard', CAST(N'1968-12-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (179, N'Chris', N'Renaud', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (180, N'Tim', N'Miller', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (181, N'David', N'Ayer', CAST(N'1968-01-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (182, N'Stephen', N'Chow', CAST(N'1962-06-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (183, N'Scott', N'Derrickson', CAST(N'1966-07-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (184, N'Joel', N'Zwick', CAST(N'1942-01-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (185, N'Fernando', N'Meirelles', CAST(N'1955-11-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (186, N'Stephen', N'Sommers', CAST(N'1962-03-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (187, N'Joe', N'Johnston', CAST(N'1950-05-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (188, N'Nancy', N'Meyers', CAST(N'1949-12-08T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (189, N'Ralph', N'Zondag', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (190, N'Kevin', N'Lima', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (191, N'Roger', N'Michell', CAST(N'1956-06-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (192, N'Peter', N'Farrelly', CAST(N'1956-12-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (193, N'Barry', N'Cook', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (194, N'Betty', N'Thomas', CAST(N'1947-07-27T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (195, N'John', N'Madden', CAST(N'1949-04-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (196, N'P. J.', N'Hogan', CAST(N'1962-11-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (197, N'Peter', N'Cattaneo', CAST(N'1964-07-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (198, N'Jan', N'de Bont', CAST(N'1943-10-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (199, N'Gary', N'Trousdale', CAST(N'1960-06-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (200, N'Stephen', N'Herek', CAST(N'1958-11-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (201, N'Cameron', N'Crowe', CAST(N'1957-07-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (202, N'Chuck', N'Russell', CAST(N'1958-05-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (203, N'Mike', N'Gabriel', CAST(N'1954-11-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (204, N'Joel', N'Schumacher', CAST(N'1939-08-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (205, N'Brad', N'Silberling', CAST(N'1963-09-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (208, N'Brian', N'Levant', CAST(N'1952-08-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (209, N'Neil', N'Jordan', CAST(N'1950-02-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (210, N'Sydney', N'Pollack', CAST(N'1934-07-01T00:00:00.000' AS DateTime), CAST(N'2008-05-26T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (211, N'Adrian', N'Lyne', CAST(N'1941-03-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (212, N'Nora', N'Ephron', CAST(N'1941-05-19T00:00:00.000' AS DateTime), CAST(N'2012-06-26T00:00:00.000' AS DateTime), N'Female');

INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (213, N'Jonathan', N'Demme', CAST(N'1944-02-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (214, N'Alan J.', N'Pakula', CAST(N'1928-04-07T00:00:00.000' AS DateTime), CAST(N'1998-11-19T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (215, N'Ron', N'Clements', CAST(N'1953-04-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (216, N'Mick', N'Jackson', CAST(N'1943-10-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (217, N'Rob', N'Reiner', CAST(N'1947-03-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (218, N'Emile', N'Ardolino', CAST(N'1943-05-09T00:00:00.000' AS DateTime), CAST(N'1993-11-20T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (219, N'Francis Ford', N'Coppola', CAST(N'1939-04-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (220, N'Penelope', N'Spheeris', CAST(N'1945-12-02T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (221, N'Oliver', N'Stone', CAST(N'1946-09-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (222, N'Jim', N'Abrahams', CAST(N'1944-05-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (223, N'Ron', N'Underwood', CAST(N'1953-11-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (224, N'Jerry', N'Zucker', CAST(N'1950-03-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (225, N'Garry', N'Marshall', CAST(N'1934-11-13T00:00:00.000' AS DateTime), CAST(N'2016-07-19T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (226, N'Steve', N'Barron', CAST(N'1956-05-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (227, N'Amy', N'Heckerling', CAST(N'1954-05-07T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (228, N'Barry', N'Levinson', CAST(N'1942-04-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (229, N'John', N'Landis', CAST(N'1950-08-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (230, N'Penny', N'Marshall', CAST(N'1943-10-15T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (231, N'John', N'Cornell', CAST(N'1941-03-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (232, N'David', N'Zucker', CAST(N'1947-10-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (233, N'Roger', N'Donaldson', CAST(N'1945-11-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (234, N'Norman', N'Jewison', CAST(N'1926-07-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (235, N'Herbert', N'Ross', CAST(N'1927-05-13T00:00:00.000' AS DateTime), CAST(N'2001-10-09T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (236, N'John', N'Badham', CAST(N'1939-08-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (237, N'Peter', N'Faiman', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (238, N'John G.', N'Avildsen', CAST(N'1935-12-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (239, N'Michael', N'Ritchie', CAST(N'1938-11-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (240, N'John', N'Hughes', CAST(N'1950-02-18T00:00:00.000' AS DateTime), CAST(N'2009-08-06T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (241, N'George P.', N'Cosmatos', CAST(N'1941-01-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (242, N'Sylvester', N'Stallone', CAST(N'1946-07-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (243, N'Lewis', N'Teague', CAST(N'1938-03-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (244, N'Martin', N'Brest', CAST(N'1951-08-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (245, N'Joe', N'Dante', CAST(N'1946-11-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (246, N'Hugh', N'Wilson', CAST(N'1943-08-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (247, N'James L.', N'Brooks', CAST(N'1940-05-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (248, N'Taylor', N'Hackford', CAST(N'1944-12-31T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (249, N'Bob', N'Clark', CAST(N'1939-08-05T00:00:00.000' AS DateTime), CAST(N'2007-04-04T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (250, N'Walter', N'Hill', CAST(N'1942-01-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (251, N'Tobe', N'Hooper', CAST(N'1943-01-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (252, N'John', N'Huston', CAST(N'1906-08-05T00:00:00.000' AS DateTime), CAST(N'1987-08-28T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (253, N'Steve', N'Gordon', NULL, CAST(N'1982-11-27T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (254, N'Hal', N'Needham', CAST(N'1931-03-06T00:00:00.000' AS DateTime), CAST(N'2013-10-25T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (255, N'Hugh', N'Hudson', CAST(N'1936-08-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (256, N'Terry', N'Gilliam', CAST(N'1940-11-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (257, N'Colin', N'Higgins', CAST(N'1941-07-28T00:00:00.000' AS DateTime), CAST(N'1988-08-05T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (258, N'Sidney', N'Poitier', CAST(N'1927-02-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (259, N'Buddy', N'Van Horn', CAST(N'1929-08-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (260, N'Howard', N'Zieff', CAST(N'1927-10-21T00:00:00.000' AS DateTime), CAST(N'2009-02-22T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (261, N'Robert', N'Benton', CAST(N'1932-09-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (262, N'Stuart', N'Rosenberg', CAST(N'1927-08-11T00:00:00.000' AS DateTime), CAST(N'2007-03-15T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (263, N'Blake', N'Edwards', CAST(N'1922-07-26T00:00:00.000' AS DateTime), CAST(N'2010-12-15T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (264, N'Carl', N'Reiner', CAST(N'1922-03-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (265, N'James', N'Frawley', CAST(N'1936-09-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (266, N'Randal', N'Kleiser', CAST(N'1946-07-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (267, N'James', N'Fargo', CAST(N'1938-08-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (268, N'Jeannot', N'Szwarc', CAST(N'1937-11-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (269, N'George A.', N'Romero', CAST(N'1940-02-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (270, N'Michael', N'Cimino', CAST(N'1939-02-03T00:00:00.000' AS DateTime), CAST(N'2016-07-02T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (271, N'Peter', N'Yates', CAST(N'1929-07-24T00:00:00.000' AS DateTime), CAST(N'2011-01-09T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (272, N'Woody', N'Allen', CAST(N'1935-12-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (273, N'Arthur', N'Hiller', CAST(N'1923-11-22T00:00:00.000' AS DateTime), CAST(N'2016-08-17T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (274, N'Jack', N'Smight', CAST(N'1925-03-09T00:00:00.000' AS DateTime), CAST(N'2003-09-01T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (275, N'Jim', N'Sharman', CAST(N'1945-03-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (276, N'Milos', N'Forman', CAST(N'1932-02-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (277, N'Sidney', N'Lumet', CAST(N'1924-06-25T00:00:00.000' AS DateTime), CAST(N'2011-04-09T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (278, N'Hal', N'Ashby', CAST(N'1929-09-02T00:00:00.000' AS DateTime), CAST(N'1988-12-27T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (279, N'Mel', N'Brooks', CAST(N'1926-06-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (281, N'Mark', N'Robson', CAST(N'1913-12-04T00:00:00.000' AS DateTime), CAST(N'1978-06-20T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (282, N'Robert', N'Aldrich', CAST(N'1918-08-09T00:00:00.000' AS DateTime), CAST(N'1983-12-05T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (283, N'George Roy', N'Hill', CAST(N'1921-12-20T00:00:00.000' AS DateTime), CAST(N'2002-12-27T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (284, N'William', N'Friedkin', CAST(N'1935-08-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (285, N'Franklin J.', N'Schaffner', CAST(N'1920-05-30T00:00:00.000' AS DateTime), CAST(N'1989-07-02T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (287, N'Wolfgang', N'Reitherman', CAST(N'1909-06-26T00:00:00.000' AS DateTime), CAST(N'1985-05-22T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (288, N'Ronald', N'Neame', CAST(N'1911-04-23T00:00:00.000' AS DateTime), CAST(N'2010-06-16T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (289, N'John', N'Boorman', CAST(N'1933-01-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (290, N'Bob', N'Fosse', CAST(N'1927-06-23T00:00:00.000' AS DateTime), CAST(N'1987-09-23T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (291, N'Don', N'Siegel', CAST(N'1912-10-26T00:00:00.000' AS DateTime), CAST(N'1991-04-20T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (292, N'Stanley', N'Kubrick', CAST(N'1928-07-26T00:00:00.000' AS DateTime), CAST(N'1999-03-07T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (293, N'Robert', N'Stevenson', CAST(N'1905-03-31T00:00:00.000' AS DateTime), CAST(N'1986-04-30T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (294, N'George', N'Seaton', CAST(N'1911-04-17T00:00:00.000' AS DateTime), CAST(N'1979-07-28T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (295, N'Robert', N'Altman', CAST(N'1925-02-20T00:00:00.000' AS DateTime), CAST(N'2006-11-20T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (297, N'Richard', N'Fleischer', CAST(N'1916-12-08T00:00:00.000' AS DateTime), CAST(N'2006-03-25T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (298, N'John', N'Schlesinger', CAST(N'1926-02-16T00:00:00.000' AS DateTime), CAST(N'2003-07-25T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (299, N'Dennis', N'Hopper', CAST(N'1936-05-17T00:00:00.000' AS DateTime), CAST(N'2010-05-29T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (300, N'Joshua', N'Logan', CAST(N'1908-10-05T00:00:00.000' AS DateTime), CAST(N'1988-07-12T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (301, N'Henry', N'Hathaway', CAST(N'1898-03-13T00:00:00.000' AS DateTime), CAST(N'1985-02-11T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (302, N'Gene', N'Saks', CAST(N'1921-11-08T00:00:00.000' AS DateTime), CAST(N'2015-03-28T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (303, N'Franco', N'Zeffirelli', CAST(N'1923-02-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (304, N'Carol', N'Reed', CAST(N'1906-12-30T00:00:00.000' AS DateTime), CAST(N'1976-04-25T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (306, N'Mike', N'Nichols', CAST(N'1931-11-06T00:00:00.000' AS DateTime), CAST(N'2014-11-19T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (307, N'Arthur', N'Penn', CAST(N'1922-09-27T00:00:00.000' AS DateTime), CAST(N'2010-09-28T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (308, N'Sergio', N'Leone', CAST(N'1929-01-03T00:00:00.000' AS DateTime), CAST(N'1989-04-30T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (309, N'John', N'Frankenheimer', CAST(N'1930-02-19T00:00:00.000' AS DateTime), CAST(N'2002-07-06T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (310, N'David', N'Lean', CAST(N'1908-03-25T00:00:00.000' AS DateTime), CAST(N'1991-04-16T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (311, N'Clive', N'Donner', CAST(N'1926-01-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (312, N'George', N'Cukor', CAST(N'1899-07-07T00:00:00.000' AS DateTime), CAST(N'1983-01-24T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (313, N'Richard', N'Lester', CAST(N'1932-01-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (314, N'Joseph L.', N'Mankiewicz', CAST(N'1909-02-11T00:00:00.000' AS DateTime), CAST(N'1993-02-05T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (315, N'John', N'Ford', CAST(N'1894-02-01T00:00:00.000' AS DateTime), CAST(N'1973-08-31T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (316, N'Stanley', N'Kramer', CAST(N'1913-09-29T00:00:00.000' AS DateTime), CAST(N'2001-02-19T00:00:00.000' AS DateTime), N'Male');

INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (317, N'Alfred', N'Hitchcock', CAST(N'1899-08-13T00:00:00.000' AS DateTime), CAST(N'1980-04-29T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (318, N'Ken', N'Annakin', CAST(N'1914-08-10T00:00:00.000' AS DateTime), CAST(N'2009-04-22T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (319, N'Lewis', N'Milestone', CAST(N'1895-09-30T00:00:00.000' AS DateTime), CAST(N'1980-09-25T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (320, N'Robert', N'Mulligan', CAST(N'1925-08-23T00:00:00.000' AS DateTime), CAST(N'2008-12-20T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (321, N'J. Lee', N'Thompson', CAST(N'1914-08-01T00:00:00.000' AS DateTime), CAST(N'2002-08-30T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (322, N'Anthony', N'Mann', CAST(N'1906-06-30T00:00:00.000' AS DateTime), CAST(N'1967-04-29T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (323, N'David', N'Swift', CAST(N'1919-07-27T00:00:00.000' AS DateTime), CAST(N'2001-12-31T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (324, N'Federico', N'Fellini', CAST(N'1920-01-20T00:00:00.000' AS DateTime), CAST(N'1993-10-31T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (325, N'John', N'Wayne', CAST(N'1907-05-26T00:00:00.000' AS DateTime), CAST(N'1979-06-11T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (326, N'William', N'Wyler', CAST(N'1902-07-01T00:00:00.000' AS DateTime), CAST(N'1981-07-27T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (327, N'Billy', N'Wilder', CAST(N'1906-06-22T00:00:00.000' AS DateTime), CAST(N'2002-03-27T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (328, N'Howard', N'Hawks', CAST(N'1896-05-30T00:00:00.000' AS DateTime), CAST(N'1977-12-26T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (329, N'Richard', N'Brooks', CAST(N'1912-05-18T00:00:00.000' AS DateTime), CAST(N'1992-03-11T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (330, N'Charles', N'Vidor', CAST(N'1900-07-27T00:00:00.000' AS DateTime), CAST(N'1959-06-04T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (331, N'John', N'Sturges', CAST(N'1910-01-03T00:00:00.000' AS DateTime), CAST(N'1992-08-18T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (332, N'King', N'Vidor', CAST(N'1894-02-08T00:00:00.000' AS DateTime), CAST(N'1982-11-01T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (333, N'Walter', N'Lang', CAST(N'1896-08-10T00:00:00.000' AS DateTime), CAST(N'1972-02-07T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (334, N'Charles', N'Walters', CAST(N'1911-11-17T00:00:00.000' AS DateTime), CAST(N'1982-08-13T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (335, N'Fred', N'Zinnemann', CAST(N'1907-04-29T00:00:00.000' AS DateTime), CAST(N'1997-03-14T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (336, N'Clyde', N'Geronimi', CAST(N'1901-06-12T00:00:00.000' AS DateTime), CAST(N'1989-04-24T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (337, N'Henry', N'Koster', CAST(N'1905-05-01T00:00:00.000' AS DateTime), CAST(N'1988-09-21T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (338, N'George', N'Stevens', CAST(N'1904-12-18T00:00:00.000' AS DateTime), CAST(N'1975-03-08T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (339, N'Richard', N'Thorpe', CAST(N'1896-02-24T00:00:00.000' AS DateTime), CAST(N'1991-05-01T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (340, N'Gene', N'Kelly', CAST(N'1912-08-23T00:00:00.000' AS DateTime), CAST(N'1996-02-02T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (341, N'Elia', N'Kazan', CAST(N'1909-09-07T00:00:00.000' AS DateTime), CAST(N'2003-09-28T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (342, N'Compton', N'Bennett', CAST(N'1900-01-15T00:00:00.000' AS DateTime), CAST(N'1974-08-11T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (344, N'Sam', N'Wood', CAST(N'1884-07-10T00:00:00.000' AS DateTime), CAST(N'1949-09-22T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (345, N'Michael', N'Curtiz', CAST(N'1888-12-24T00:00:00.000' AS DateTime), CAST(N'1962-04-10T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (346, N'Sidney', N'Lanfield', CAST(N'1898-04-20T00:00:00.000' AS DateTime), CAST(N'1972-06-20T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (347, N'Alfred L.', N'Werker', CAST(N'1896-12-02T00:00:00.000' AS DateTime), CAST(N'1975-07-28T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (348, N'Charlie', N'Chaplin', CAST(N'1889-04-16T00:00:00.000' AS DateTime), CAST(N'1977-12-25T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (349, N'Victor', N'Fleming', CAST(N'1889-02-23T00:00:00.000' AS DateTime), CAST(N'1949-01-06T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (350, N'Frank', N'Lloyd', CAST(N'1886-02-02T00:00:00.000' AS DateTime), CAST(N'1960-08-10T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (351, N'David', N'Hand', CAST(N'1900-01-23T00:00:00.000' AS DateTime), CAST(N'1986-10-11T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (352, N'Orson', N'Welles', CAST(N'1915-05-06T00:00:00.000' AS DateTime), CAST(N'1985-10-10T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (353, N'Fritz', N'Lang', CAST(N'1890-12-05T00:00:00.000' AS DateTime), CAST(N'1976-08-02T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (354, N'Clarence', N'Brown', CAST(N'1890-05-10T00:00:00.000' AS DateTime), CAST(N'1987-08-17T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (355, N'Julien', N'Duvivier', CAST(N'1896-10-08T00:00:00.000' AS DateTime), CAST(N'1967-10-29T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (356, N'James', N'Whale', CAST(N'1889-07-22T00:00:00.000' AS DateTime), CAST(N'1957-05-29T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (357, N'Tod', N'Browning', CAST(N'1880-07-12T00:00:00.000' AS DateTime), CAST(N'1962-10-06T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (358, N'William A.', N'Wellman', CAST(N'1896-02-29T00:00:00.000' AS DateTime), CAST(N'1975-12-09T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (359, N'Frank', N'Capra', CAST(N'1897-05-18T00:00:00.000' AS DateTime), CAST(N'1991-09-03T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (360, N'Robert', N'Siodmak', CAST(N'1900-08-08T00:00:00.000' AS DateTime), CAST(N'1973-03-10T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (361, N'Laurence', N'Olivier', CAST(N'1907-05-22T00:00:00.000' AS DateTime), CAST(N'1989-07-11T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (362, N'George', N'Sidney', CAST(N'1916-10-04T00:00:00.000' AS DateTime), CAST(N'2002-05-05T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (363, N'Lo', N'Wei', CAST(N'1918-12-12T00:00:00.000' AS DateTime), CAST(N'1996-01-20T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (364, N'Bruce', N'Lee', CAST(N'1940-11-27T00:00:00.000' AS DateTime), CAST(N'1973-07-20T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (365, N'Robert', N'Clouse', CAST(N'1928-03-06T00:00:00.000' AS DateTime), CAST(N'1997-02-04T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (366, N'Terry', N'Jones', CAST(N'1942-02-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (367, N'Woo-ping', N'Yuen', CAST(N'1945-01-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (368, N'Chi-hwa', N'Chen', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (369, N'Jackie', N'Chan', CAST(N'1954-04-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (370, N'Hark', N'Tsui', CAST(N'1950-02-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (371, N'Corey', N'Yuen', CAST(N'1951-02-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (372, N'Gordon', N'Chan', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (373, N'Andrzej', N'Bartkowiak', CAST(N'1950-03-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (374, N'James', N'Wong', CAST(N'1959-04-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (375, N'Chris', N'Nahon', CAST(N'1968-12-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (376, N'Louis', N'Leterrier', CAST(N'1973-06-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (377, N'Ronny', N'Yu', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (378, N'Peter', N'Chan', CAST(N'1962-11-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (379, N'Rob', N'Minkoff', CAST(N'1962-08-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (380, N'Simon', N'West', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (381, N'Patrick', N'Hughes', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (382, N'Desmond', N'Davis', CAST(N'1926-05-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (384, N'Gareth', N'Evans', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (385, N'Prachya', N'Pinkaew', CAST(N'1962-09-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (387, N'Tony', N'Jaa', CAST(N'1976-02-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (388, N'Wilson', N'Yip', CAST(N'1963-10-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (389, N'Kon', N'Ichikawa', CAST(N'1915-11-20T00:00:00.000' AS DateTime), CAST(N'2008-02-13T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (390, N'Hiroshi', N'Inagaki', CAST(N'1905-12-30T00:00:00.000' AS DateTime), CAST(N'1980-05-21T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (391, N'Carl', N'Rinsch', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (392, N'Takashi', N'Miike', CAST(N'1960-08-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (393, N'Takeshi', N'Kitano', CAST(N'1947-01-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (394, N'Kenneth', N'Branagh', CAST(N'1960-12-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (395, N'Kinji', N'Fukasaku', CAST(N'1930-07-03T00:00:00.000' AS DateTime), CAST(N'2003-01-12T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (396, N'Tom', N'Kotani', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (397, N'Yojiro', N'Takita', CAST(N'1955-12-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (398, N'Yoji', N'Yamada', CAST(N'1931-09-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (399, N'Kenji', N'Misumi', CAST(N'1921-03-02T00:00:00.000' AS DateTime), CAST(N'1975-09-24T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (400, N'Buichi', N'Saito', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (401, N'Yoshiyuki', N'Kuroda', CAST(N'1928-03-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (402, N'Ishiro', N'Honda', CAST(N'1911-05-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (403, N'Pierre', N'Morel', CAST(N'1964-05-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (404, N'Patrick', N'Alessandrin', CAST(N'1965-05-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (405, N'Jean-Francois', N'Richet', CAST(N'1966-07-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (406, N'Jacques', N'Audiard', CAST(N'1952-04-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (407, N'Shane', N'Meadows', CAST(N'1972-12-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (408, N'Steve', N'Bendelack', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (409, N'Ben', N'Wheatley', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (410, N'Stephen', N'Campanelli', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (411, N'Neill', N'Blomkamp', CAST(N'1979-09-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (412, N'Jonathan', N'English', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (413, N'Michael J.', N'Bassett', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (414, N'Brian', N'Helgeland', CAST(N'1961-01-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (415, N'Juan Carlos', N'Fresnadillo', CAST(N'1967-12-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (416, N'Henry', N'Hobson', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (417, N'Mikael', N'Hafstrom', CAST(N'1960-07-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (418, N'Kim', N'Jee-woon', CAST(N'1964-07-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (419, N'McG', NULL, CAST(N'1968-08-09T00:00:00.000' AS DateTime), NULL, N'Male');

INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (420, N'Peter', N'Hyams', CAST(N'1943-07-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (421, N'John', N'Irvin', CAST(N'1940-05-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (422, N'Mark L.', N'Lester', CAST(N'1946-11-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (423, N'John', N'Milius', CAST(N'1944-04-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (424, N'Marcus', N'Nispel', CAST(N'1963-05-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (425, N'George', N'Butler', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (426, N'Arthur Allan', N'Seidelman', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (427, N'Matthew', N'Vaughn', CAST(N'1971-03-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (428, N'Patrick', N'Tatopoulos', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (429, N'Mans', N'Marlind', CAST(N'1969-07-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (430, N'Anna', N'Foerster', NULL, NULL, N'Female');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (431, N'Angelina', N'Jolie', CAST(N'1975-06-04T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (432, N'Alexander', N'Mackendrick', CAST(N'1912-09-08T00:00:00.000' AS DateTime), CAST(N'1993-12-22T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (433, N'Howard', N'McCain', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (435, N'Sheldon', N'Lettich', CAST(N'1951-01-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (436, N'Deran', N'Sarafian', CAST(N'1958-01-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (437, N'Mark', N'DiSalle', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (438, N'Albert', N'Pyun', CAST(N'1953-05-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (439, N'Newt', N'Arnold', CAST(N'1922-02-22T00:00:00.000' AS DateTime), CAST(N'2000-02-12T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (440, N'Antoine', N'Fuqua', CAST(N'1966-01-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (441, N'Mike', N'Hodges', CAST(N'1932-07-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (442, N'Lee', N'Sang-il', CAST(N'1974-01-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (443, N'Ted', N'Post', CAST(N'1918-03-31T00:00:00.000' AS DateTime), CAST(N'2013-08-20T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (444, N'Brian G.', N'Hutton', CAST(N'1935-01-01T00:00:00.000' AS DateTime), CAST(N'2014-08-19T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (445, N'David S.', N'Goyer', CAST(N'1965-12-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (446, N'Marco', N'Brambilla', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (447, N'Kevin', N'Hooks', CAST(N'1958-09-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (448, N'Sam', N'Peckinpah', CAST(N'1925-02-21T00:00:00.000' AS DateTime), CAST(N'1984-12-28T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (449, N'Tony', N'Kaye', CAST(N'1952-07-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (450, N'Cy', N'Endfield', CAST(N'1914-11-10T00:00:00.000' AS DateTime), CAST(N'1995-04-16T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (451, N'Paul', N'McGuigan', CAST(N'1963-09-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (452, N'Jose', N'Padilha', CAST(N'1967-08-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (453, N'Kevin', N'Smith', CAST(N'1970-08-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (454, N'Nicolas', N'Winding Refn', CAST(N'1970-09-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (455, N'Ben', N'Stiller', CAST(N'1965-11-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (456, N'Jim', N'Henson', CAST(N'1936-09-24T00:00:00.000' AS DateTime), CAST(N'1990-05-16T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (457, N'David', N'Lynch', CAST(N'1946-01-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (458, N'Paul', N'Feig', CAST(N'1962-09-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (459, N'Conrad', N'Vernon', CAST(N'1968-07-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (460, N'Peter', N'Hewitt', CAST(N'1962-10-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (461, N'Terry', N'Zwigoff', CAST(N'1949-05-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (462, N'Mark', N'Waters', CAST(N'1964-06-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (463, N'Gareth', N'Edwards', CAST(N'1975-06-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (464, N'Morten', N'Tyldum', CAST(N'1967-05-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (465, N'Alex', N'Garland', CAST(N'1970-05-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (466, N'Joe', N'Wright', CAST(N'1972-08-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (467, N'Bernard', N'Rose', CAST(N'1960-08-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (468, N'Lasse', N'Hallstrom', CAST(N'1946-06-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (469, N'Anthony', N'Minghella', CAST(N'1954-01-06T00:00:00.000' AS DateTime), CAST(N'2008-03-18T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (470, N'Kathryn', N'Bigelow', CAST(N'1951-11-27T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (471, N'Ericson', N'Core', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (472, N'Catherine', N'Hardwicke', CAST(N'1955-10-21T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (473, N'Niels Arden', N'Oplev', CAST(N'1961-03-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (474, N'Daniel', N'Alfredson', CAST(N'1959-05-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (475, N'Chad', N'Stahelski', CAST(N'1968-09-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (476, N'Ingmar', N'Bergman', CAST(N'1918-07-14T00:00:00.000' AS DateTime), CAST(N'2007-07-30T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (477, N'Alejandro G.', N'Inarritu', CAST(N'1963-08-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (478, N'Seth', N'MacFarlane', CAST(N'1973-10-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (479, N'Tommy', N'Wirkola', CAST(N'1979-12-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (480, N'Oliver', N'Owl', CAST(N'1989-12-31T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (481, N'Wendy', N'Wise', CAST(N'1989-12-31T00:11:00.000' AS DateTime), NULL, N'Female');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (482, N'Paul Thomas', N'Anderson', CAST(N'1970-06-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (483, N'Jonathan', N'Hensleigh', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (484, N'James', N'Marsh', CAST(N'1963-04-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (485, N'Rian', N'Johnson', CAST(N'1973-12-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (486, N'Steve', N'McQueen', CAST(N'1969-10-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (487, N'Denis', N'Villeneuve', CAST(N'1967-10-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (488, N'James', N'Mangold', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (489, N'Tom', N'Hooper', CAST(N'1972-10-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (490, N'Christopher', N'Smith', CAST(N'1972-07-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (491, N'Justin', N'Chadwick', CAST(N'1968-12-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (492, N'George', N'Nolfi', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (493, N'Peyton', N'Reed', CAST(N'1964-07-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (494, N'Delmer', N'Daves', CAST(N'1904-07-24T00:00:00.000' AS DateTime), CAST(N'1977-08-17T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (495, N'Christopher', N'Cain', CAST(N'1943-10-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (496, N'Geoff', N'Murphy', CAST(N'1938-10-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (497, N'Phil', N'Lord', CAST(N'1975-07-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (498, N'Christopher', N'Miller', CAST(N'1975-09-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (499, N'Anatole', N'Litvak', CAST(N'1902-05-21T00:00:00.000' AS DateTime), CAST(N'1974-12-15T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (500, N'Akiva', N'Schaffer', CAST(N'1977-12-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (501, N'Cody', N'Cameron', CAST(N'1970-10-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (502, N'Kevin', N'Macdonald', CAST(N'1967-10-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (503, N'Neil', N'Marshall', CAST(N'1970-05-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (504, N'Don', N'Chaffey', CAST(N'1917-08-05T00:00:00.000' AS DateTime), CAST(N'1990-11-13T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (505, N'David', N'Lowery', NULL, NULL, N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (506, N'Stuart', N'Burge', CAST(N'1918-01-15T00:00:00.000' AS DateTime), CAST(N'2002-01-24T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (507, N'Nicholas', N'Ray', CAST(N'1911-08-07T00:00:00.000' AS DateTime), CAST(N'1979-06-16T00:00:00.000' AS DateTime), N'Male');
INSERT Director (DirectorID, FirstName, FamilyName, DoB, DoD, Gender) VALUES (508, N'Les', N'Mayfield', NULL, NULL, N'Male');

INSERT Genre (GenreID, Genre) VALUES (1, N'Western');
INSERT Genre (GenreID, Genre) VALUES (2, N'Drama');
INSERT Genre (GenreID, Genre) VALUES (3, N'Science Fiction');
INSERT Genre (GenreID, Genre) VALUES (4, N'Action');
INSERT Genre (GenreID, Genre) VALUES (5, N'War');
INSERT Genre (GenreID, Genre) VALUES (6, N'Romance');
INSERT Genre (GenreID, Genre) VALUES (7, N'Comedy');
INSERT Genre (GenreID, Genre) VALUES (8, N'Adventure');
INSERT Genre (GenreID, Genre) VALUES (9, N'Thriller');
INSERT Genre (GenreID, Genre) VALUES (10, N'Horror');
INSERT Genre (GenreID, Genre) VALUES (11, N'Animation');
INSERT Genre (GenreID, Genre) VALUES (12, N'Fantasy');
INSERT Genre (GenreID, Genre) VALUES (13, N'History');
INSERT Genre (GenreID, Genre) VALUES (14, N'Crime');
INSERT Genre (GenreID, Genre) VALUES (17, N'Martial Arts');
INSERT Genre (GenreID, Genre) VALUES (18, N'Mystery');
INSERT Genre (GenreID, Genre) VALUES (19, N'Biography');
INSERT Genre (GenreID, Genre) VALUES (20, N'Musical');
INSERT Genre (GenreID, Genre) VALUES (22, N'Sport');
INSERT Genre (GenreID, Genre) VALUES (24, N'Documentary');
INSERT Genre (GenreID, Genre) VALUES (25, N'Disaster');
INSERT Genre (GenreID, Genre) VALUES (26, N'Family');
INSERT Genre (GenreID, Genre) VALUES (27, N'Romantic Comedy');
INSERT Genre (GenreID, Genre) VALUES (28, N'Awful');


INSERT Language (LanguageID, Language) VALUES (1, N'English');
INSERT Language (LanguageID, Language) VALUES (2, N'Cantonese');
INSERT Language (LanguageID, Language) VALUES (3, N'French');
INSERT Language (LanguageID, Language) VALUES (4, N'German');
INSERT Language (LanguageID, Language) VALUES (5, N'Mandarin');
INSERT Language (LanguageID, Language) VALUES (6, N'Japanese');
INSERT Language (LanguageID, Language) VALUES (7, N'Aramaic');
INSERT Language (LanguageID, Language) VALUES (8, N'Mayan');
INSERT Language (LanguageID, Language) VALUES (9, N'Russian');
INSERT Language (LanguageID, Language) VALUES (10, N'Silent');
INSERT Language (LanguageID, Language) VALUES (11, N'Portugese');
INSERT Language (LanguageID, Language) VALUES (12, N'Italian');
INSERT Language (LanguageID, Language) VALUES (13, N'Indonesian');
INSERT Language (LanguageID, Language) VALUES (14, N'Thai');
INSERT Language (LanguageID, Language) VALUES (15, N'Swedish');

INSERT Studio (StudioID, Studio) VALUES (1, N'Universal Pictures');
INSERT Studio (StudioID, Studio) VALUES (2, N'Paramount Pictures');
INSERT Studio (StudioID, Studio) VALUES (3, N'Walt Disney Pictures');
INSERT Studio (StudioID, Studio) VALUES (4, N'20th Century Fox');
INSERT Studio (StudioID, Studio) VALUES (5, N'Columbia Pictures');
INSERT Studio (StudioID, Studio) VALUES (6, N'Warner Bros. Pictures');
INSERT Studio (StudioID, Studio) VALUES (7, N'Touchstone Pictures');
INSERT Studio (StudioID, Studio) VALUES (8, N'Dreamworks');
INSERT Studio (StudioID, Studio) VALUES (9, N'ImageMovers');
INSERT Studio (StudioID, Studio) VALUES (10, N'Pixar Animation Studios');
INSERT Studio (StudioID, Studio) VALUES (11, N'Metro-ldwyn-Mayer');
INSERT Studio (StudioID, Studio) VALUES (12, N'Chris Lee Productions');
INSERT Studio (StudioID, Studio) VALUES (13, N'Lucasfilm');
INSERT Studio (StudioID, Studio) VALUES (14, N'New Line Cinema');
INSERT Studio (StudioID, Studio) VALUES (15, N'Carolco Pictures');
INSERT Studio (StudioID, Studio) VALUES (16, N'Miramax Films');
INSERT Studio (StudioID, Studio) VALUES (17, N'Jerry Bruckheimer Films');
INSERT Studio (StudioID, Studio) VALUES (18, N'Gaumont');
INSERT Studio (StudioID, Studio) VALUES (19, N'Revolution Studios');
INSERT Studio (StudioID, Studio) VALUES (20, N'Imagine Entertainment');
INSERT Studio (StudioID, Studio) VALUES (21, N'Morgan Creek Productions');
INSERT Studio (StudioID, Studio) VALUES (22, N'United Artists');
INSERT Studio (StudioID, Studio) VALUES (23, N'Dimension Films');
INSERT Studio (StudioID, Studio) VALUES (24, N'Braveworld Productions');
INSERT Studio (StudioID, Studio) VALUES (25, N'Icon Productions');
INSERT Studio (StudioID, Studio) VALUES (26, N'Carolina Bank');
INSERT Studio (StudioID, Studio) VALUES (27, N'Tig Productions');
INSERT Studio (StudioID, Studio) VALUES (28, N'Dune Entertainment');
INSERT Studio (StudioID, Studio) VALUES (29, N'Beijing New Picture Film Co.');
INSERT Studio (StudioID, Studio) VALUES (30, N'Big Talk Productions');
INSERT Studio (StudioID, Studio) VALUES (31, N'The Malpaso Company');
INSERT Studio (StudioID, Studio) VALUES (32, N'British Film Council');
INSERT Studio (StudioID, Studio) VALUES (33, N'Channel Four Films');
INSERT Studio (StudioID, Studio) VALUES (34, N'Constantin Film');
INSERT Studio (StudioID, Studio) VALUES (35, N'Bavaria Film');
INSERT Studio (StudioID, Studio) VALUES (36, N'Bayerischer Rundfunk');
INSERT Studio (StudioID, Studio) VALUES (37, N'RKO Radio Pictures');
INSERT Studio (StudioID, Studio) VALUES (38, N'Toho Company');
INSERT Studio (StudioID, Studio) VALUES (39, N'Kathbur Pictures');
INSERT Studio (StudioID, Studio) VALUES (40, N'Avalon Studios');
INSERT Studio (StudioID, Studio) VALUES (41, N'Amblin Entertainment');
INSERT Studio (StudioID, Studio) VALUES (42, N'WingNut Films');
INSERT Studio (StudioID, Studio) VALUES (43, N'Legendary Pictures');
INSERT Studio (StudioID, Studio) VALUES (44, N'Spyglass Entertainment');
INSERT Studio (StudioID, Studio) VALUES (45, N'rdon Company');
INSERT Studio (StudioID, Studio) VALUES (46, N'di Bonaventura Pictures');
INSERT Studio (StudioID, Studio) VALUES (47, N'Heyday Films');
INSERT Studio (StudioID, Studio) VALUES (48, N'Shangri-La Entertainment');
INSERT Studio (StudioID, Studio) VALUES (49, N'Village Roadshow Pictures');
INSERT Studio (StudioID, Studio) VALUES (50, N'Helena Productions');
INSERT Studio (StudioID, Studio) VALUES (51, N'Eon Productions');
INSERT Studio (StudioID, Studio) VALUES (52, N'Silver Pictures');
INSERT Studio (StudioID, Studio) VALUES (53, N'Marvel Entertainment');
INSERT Studio (StudioID, Studio) VALUES (54, N'Original Film');
INSERT Studio (StudioID, Studio) VALUES (55, N'Square Pictures');
INSERT Studio (StudioID, Studio) VALUES (56, N'The Kennedy Marshall Company');
INSERT Studio (StudioID, Studio) VALUES (57, N'Centropolis Entertainment');
INSERT Studio (StudioID, Studio) VALUES (58, N'Radar Pictures');
INSERT Studio (StudioID, Studio) VALUES (59, N'Cruise Wagner Productions');
INSERT Studio (StudioID, Studio) VALUES (60, N'Lion Rock Productions');
INSERT Studio (StudioID, Studio) VALUES (61, N'Jerry Weintraub Productions');
INSERT Studio (StudioID, Studio) VALUES (62, N'Cheyenne Enterprises');
INSERT Studio (StudioID, Studio) VALUES (63, N'Walden Media');
INSERT Studio (StudioID, Studio) VALUES (64, N'Scott Free Productions');
INSERT Studio (StudioID, Studio) VALUES (65, N'Regency Enterprises');
INSERT Studio (StudioID, Studio) VALUES (66, N'Forward Pass');
INSERT Studio (StudioID, Studio) VALUES (67, N'Peters Entertainment');
INSERT Studio (StudioID, Studio) VALUES (68, N'Davis Entertainment');
INSERT Studio (StudioID, Studio) VALUES (69, N'The Bedford Falls Company');
INSERT Studio (StudioID, Studio) VALUES (70, N'Lightstorm Entertainment');
INSERT Studio (StudioID, Studio) VALUES (71, N'Roger Birnbaum Productions');
INSERT Studio (StudioID, Studio) VALUES (72, N'Cinergi Pictures');
INSERT Studio (StudioID, Studio) VALUES (73, N'The Zanuck Company');
INSERT Studio (StudioID, Studio) VALUES (74, N'Strike Entertainment');
INSERT Studio (StudioID, Studio) VALUES (75, N'Hollywood Pictures');
INSERT Studio (StudioID, Studio) VALUES (76, N'Apatow Productions');
INSERT Studio (StudioID, Studio) VALUES (77, N'Gracie Films');
INSERT Studio (StudioID, Studio) VALUES (78, N'Mandalay Pictures');
INSERT Studio (StudioID, Studio) VALUES (79, N'Fox 2000 Pictures');
INSERT Studio (StudioID, Studio) VALUES (80, N'Millennium Films');
INSERT Studio (StudioID, Studio) VALUES (81, N'Scott Rudin Productions');
INSERT Studio (StudioID, Studio) VALUES (82, N'Weed Road Pictures');
INSERT Studio (StudioID, Studio) VALUES (83, N'Dovemead');
INSERT Studio (StudioID, Studio) VALUES (84, N'A Band Apart');
INSERT Studio (StudioID, Studio) VALUES (85, N'Brandywine Productions');
INSERT Studio (StudioID, Studio) VALUES (86, N'Pacific Data Images');
INSERT Studio (StudioID, Studio) VALUES (87, N'Tapestry Films');
INSERT Studio (StudioID, Studio) VALUES (88, N'Taliafilm');
INSERT Studio (StudioID, Studio) VALUES (89, N'Jersey Films');
INSERT Studio (StudioID, Studio) VALUES (90, N'Virtual Studios');
INSERT Studio (StudioID, Studio) VALUES (91, N'Lakeshore Entertainment');
INSERT Studio (StudioID, Studio) VALUES (92, N'Mace Neufeld Productions');
INSERT Studio (StudioID, Studio) VALUES (93, N'Black Rhino');
INSERT Studio (StudioID, Studio) VALUES (94, N'The Ladd Company');
INSERT Studio (StudioID, Studio) VALUES (95, N'Republic Pictures');
INSERT Studio (StudioID, Studio) VALUES (96, N'Dino De Laurentiis Corporation');
INSERT Studio (StudioID, Studio) VALUES (97, N'ldcrest Films');
INSERT Studio (StudioID, Studio) VALUES (98, N'Jinks/Cohen Company');
INSERT Studio (StudioID, Studio) VALUES (99, N'EDKO Films');
INSERT Studio (StudioID, Studio) VALUES (100, N'River Road Entertainment');
INSERT Studio (StudioID, Studio) VALUES (101, N'Newmarket Capital Group');
INSERT Studio (StudioID, Studio) VALUES (102, N'UGC');
INSERT Studio (StudioID, Studio) VALUES (103, N'lden Harvest');
INSERT Studio (StudioID, Studio) VALUES (104, N'Polygram Filmed Entertainment');
INSERT Studio (StudioID, Studio) VALUES (105, N'2929 Entertainment');
INSERT Studio (StudioID, Studio) VALUES (106, N'Hemdale');
INSERT Studio (StudioID, Studio) VALUES (107, N'Bad Hat Harry Productions');
INSERT Studio (StudioID, Studio) VALUES (108, N'Summit Entertainment');
INSERT Studio (StudioID, Studio) VALUES (109, N'Channel One Russia');
INSERT Studio (StudioID, Studio) VALUES (110, N'Live Entertainment');
INSERT Studio (StudioID, Studio) VALUES (111, N'The Con');
INSERT Studio (StudioID, Studio) VALUES (112, N'Greenwich Film Productions');
INSERT Studio (StudioID, Studio) VALUES (113, N'DNA Films');
INSERT Studio (StudioID, Studio) VALUES (114, N'Communicado Productions');
INSERT Studio (StudioID, Studio) VALUES (115, N'Aardman Animations');
INSERT Studio (StudioID, Studio) VALUES (116, N'Jesse L. Lasky Feature Play Company');
INSERT Studio (StudioID, Studio) VALUES (117, N'Thanhouser Film Corporation');
INSERT Studio (StudioID, Studio) VALUES (118, N'Fox Film Corporation');
INSERT Studio (StudioID, Studio) VALUES (119, N'Keystone Film Company');
INSERT Studio (StudioID, Studio) VALUES (120, N'Color Force');
INSERT Studio (StudioID, Studio) VALUES (121, N'Famous Players Film Company');
INSERT Studio (StudioID, Studio) VALUES (122, N'Deutsche Bioscop');
INSERT Studio (StudioID, Studio) VALUES (123, N'Chernin Entertainment');
INSERT Studio (StudioID, Studio) VALUES (124, N'Illumination Entertainment');
INSERT Studio (StudioID, Studio) VALUES (125, N'Temple Hill Entertainment');
INSERT Studio (StudioID, Studio) VALUES (126, N'Bad Robot Productions');
INSERT Studio (StudioID, Studio) VALUES (127, N'Sony Pictures');
INSERT Studio (StudioID, Studio) VALUES (128, N'Relativity Media');
INSERT Studio (StudioID, Studio) VALUES (129, N'Playtone Entertainment');
INSERT Studio (StudioID, Studio) VALUES (130, N'Overbrook Entertainment');
INSERT Studio (StudioID, Studio) VALUES (131, N'Tribeca Productions');
INSERT Studio (StudioID, Studio) VALUES (132, N'Intermedia');
INSERT Studio (StudioID, Studio) VALUES (133, N'DC Entertainment');
INSERT Studio (StudioID, Studio) VALUES (134, N'China Film Group');
INSERT Studio (StudioID, Studio) VALUES (135, N'ld Circle Films');
INSERT Studio (StudioID, Studio) VALUES (136, N'O2 Filmes');
INSERT Studio (StudioID, Studio) VALUES (137, N'Alphaville Films');
INSERT Studio (StudioID, Studio) VALUES (138, N'Eric''s Boy');
INSERT Studio (StudioID, Studio) VALUES (139, N'Zucker Brothers Productions');
INSERT Studio (StudioID, Studio) VALUES (140, N'Kopelson Entertainment');
INSERT Studio (StudioID, Studio) VALUES (141, N'Interscope Communications');
INSERT Studio (StudioID, Studio) VALUES (142, N'Dark Horse Entertainment');
INSERT Studio (StudioID, Studio) VALUES (143, N'Mark rdon Productions');
INSERT Studio (StudioID, Studio) VALUES (144, N'Motion Picture Corporation of America');
INSERT Studio (StudioID, Studio) VALUES (145, N'Geffen Pictures');
INSERT Studio (StudioID, Studio) VALUES (146, N'Blue Wolf Productions');
INSERT Studio (StudioID, Studio) VALUES (147, N'TriStar Pictures');
INSERT Studio (StudioID, Studio) VALUES (148, N'Hughes Entertainment');
INSERT Studio (StudioID, Studio) VALUES (149, N'Castle Rock Entertainment');
INSERT Studio (StudioID, Studio) VALUES (150, N'Strong Heart/Demme Production');
INSERT Studio (StudioID, Studio) VALUES (151, N'Orion Pictures');
INSERT Studio (StudioID, Studio) VALUES (152, N'Eddie Murphy Productions');
INSERT Studio (StudioID, Studio) VALUES (153, N'Rastar');
INSERT Studio (StudioID, Studio) VALUES (154, N'Kennedy Miller');
INSERT Studio (StudioID, Studio) VALUES (155, N'Rimfire Films');
INSERT Studio (StudioID, Studio) VALUES (156, N'Mirage Studios');
INSERT Studio (StudioID, Studio) VALUES (157, N'Lorimar Productions');
INSERT Studio (StudioID, Studio) VALUES (158, N'Astral Media');
INSERT Studio (StudioID, Studio) VALUES (159, N'Handmade Films');
INSERT Studio (StudioID, Studio) VALUES (160, N'Zoetrope Studios');
INSERT Studio (StudioID, Studio) VALUES (161, N'Henson Associates');
INSERT Studio (StudioID, Studio) VALUES (162, N'Laurel Group');
INSERT Studio (StudioID, Studio) VALUES (163, N'Sellers-Edwards Productions');
INSERT Studio (StudioID, Studio) VALUES (164, N'EMI Films');
INSERT Studio (StudioID, Studio) VALUES (165, N'Joseph E. Levine Productions');
INSERT Studio (StudioID, Studio) VALUES (166, N'Allied Artists');
INSERT Studio (StudioID, Studio) VALUES (167, N'The Mirisch Production Company');
INSERT Studio (StudioID, Studio) VALUES (168, N'Hawk Films');
INSERT Studio (StudioID, Studio) VALUES (169, N'Alfred J. Hitchcock Productions');
INSERT Studio (StudioID, Studio) VALUES (170, N'Horizon Pictures');
INSERT Studio (StudioID, Studio) VALUES (171, N'Samuel Bronston Productions');
INSERT Studio (StudioID, Studio) VALUES (172, N'Pathe');
INSERT Studio (StudioID, Studio) VALUES (173, N'Bryna Productions');
INSERT Studio (StudioID, Studio) VALUES (174, N'Shamley Productions');
INSERT Studio (StudioID, Studio) VALUES (175, N'Selznick International Pictures');
INSERT Studio (StudioID, Studio) VALUES (176, N'Bing Crosby Productions');
INSERT Studio (StudioID, Studio) VALUES (177, N'Romulus Films');
INSERT Studio (StudioID, Studio) VALUES (178, N'Transatlantic Pictures');
INSERT Studio (StudioID, Studio) VALUES (179, N'Cosmopolitan Productions');
INSERT Studio (StudioID, Studio) VALUES (180, N'Charles Chaplin Film Corporation');
INSERT Studio (StudioID, Studio) VALUES (181, N'Seven Arts Productions');
INSERT Studio (StudioID, Studio) VALUES (182, N'Mercury Productions');
INSERT Studio (StudioID, Studio) VALUES (183, N'Universum Film');
INSERT Studio (StudioID, Studio) VALUES (184, N'London Film Productions');
INSERT Studio (StudioID, Studio) VALUES (185, N'Liberty Films');
INSERT Studio (StudioID, Studio) VALUES (186, N'Cineguild');
INSERT Studio (StudioID, Studio) VALUES (187, N'Ortus Films');
INSERT Studio (StudioID, Studio) VALUES (188, N'Two Cities Films');
INSERT Studio (StudioID, Studio) VALUES (189, N'Sierra Pictures');
INSERT Studio (StudioID, Studio) VALUES (190, N'Python (Monty) Pictures');
INSERT Studio (StudioID, Studio) VALUES (191, N'Seasonal Film Corporation');
INSERT Studio (StudioID, Studio) VALUES (192, N'Lo Wei Motion Picture Company');
INSERT Studio (StudioID, Studio) VALUES (193, N'od Year Movie Company');
INSERT Studio (StudioID, Studio) VALUES (194, N'Eastern Productions');
INSERT Studio (StudioID, Studio) VALUES (195, N'StudioCanal');
INSERT Studio (StudioID, Studio) VALUES (196, N'EuropaCorp');
INSERT Studio (StudioID, Studio) VALUES (197, N'We Pictures');
INSERT Studio (StudioID, Studio) VALUES (198, N'Cinema City');
INSERT Studio (StudioID, Studio) VALUES (199, N'lden Princess Film Production Limited');
INSERT Studio (StudioID, Studio) VALUES (200, N'Film Workshop');
INSERT Studio (StudioID, Studio) VALUES (201, N'PT. Merantau Films');
INSERT Studio (StudioID, Studio) VALUES (202, N'Baa-Ram-Ewe');
INSERT Studio (StudioID, Studio) VALUES (203, N'Sahamongkol Film International');
INSERT Studio (StudioID, Studio) VALUES (204, N'Dreams Salon Entertainment');
INSERT Studio (StudioID, Studio) VALUES (205, N'Daiei Film');
INSERT Studio (StudioID, Studio) VALUES (206, N'Moving Picture Company');
INSERT Studio (StudioID, Studio) VALUES (207, N'Omega Project');
INSERT Studio (StudioID, Studio) VALUES (208, N'Recorded Picture Company');
INSERT Studio (StudioID, Studio) VALUES (209, N'Asahi National Broadcasting Company');
INSERT Studio (StudioID, Studio) VALUES (210, N'Hasbro Studios');
INSERT Studio (StudioID, Studio) VALUES (211, N'Bandai Visual Company');
INSERT Studio (StudioID, Studio) VALUES (212, N'AM Associates');
INSERT Studio (StudioID, Studio) VALUES (213, N'Toei Kyoto');
INSERT Studio (StudioID, Studio) VALUES (214, N'Rankin/Bass Productions');
INSERT Studio (StudioID, Studio) VALUES (215, N'TV Tokyo');
INSERT Studio (StudioID, Studio) VALUES (216, N'Sedic International');
INSERT Studio (StudioID, Studio) VALUES (217, N'Eisei Gekijo');
INSERT Studio (StudioID, Studio) VALUES (218, N'La Petite Reine');
INSERT Studio (StudioID, Studio) VALUES (219, N'Rogue Pictures');
INSERT Studio (StudioID, Studio) VALUES (220, N'Why Not Productions');
INSERT Studio (StudioID, Studio) VALUES (221, N'FilmFour Productions');
INSERT Studio (StudioID, Studio) VALUES (222, N'BBC Films');
INSERT Studio (StudioID, Studio) VALUES (223, N'Baby Cow Productions');
INSERT Studio (StudioID, Studio) VALUES (224, N'Azari Media');
INSERT Studio (StudioID, Studio) VALUES (225, N'Blueprint Pictures');
INSERT Studio (StudioID, Studio) VALUES (226, N'Mythic International Entertainment');
INSERT Studio (StudioID, Studio) VALUES (228, N'Lionsgate');
INSERT Studio (StudioID, Studio) VALUES (229, N'Open Road Films');
INSERT Studio (StudioID, Studio) VALUES (230, N'Beacon Pictures');
INSERT Studio (StudioID, Studio) VALUES (231, N'White Mountain Films');
INSERT Studio (StudioID, Studio) VALUES (232, N'RAF Industires');
INSERT Studio (StudioID, Studio) VALUES (234, N'Plan B Entertainment');
INSERT Studio (StudioID, Studio) VALUES (235, N'Marv Films');
INSERT Studio (StudioID, Studio) VALUES (236, N'Tim Burton Productions');
INSERT Studio (StudioID, Studio) VALUES (237, N'Ealing Studios');
INSERT Studio (StudioID, Studio) VALUES (238, N'Kings Road Entertainment');
INSERT Studio (StudioID, Studio) VALUES (239, N'The Cannon Group');
INSERT Studio (StudioID, Studio) VALUES (240, N'Stanley Kubrick Productions');
INSERT Studio (StudioID, Studio) VALUES (241, N'Nikkatsu');
INSERT Studio (StudioID, Studio) VALUES (242, N'Act III Communications');
INSERT Studio (StudioID, Studio) VALUES (243, N'First Artists');
INSERT Studio (StudioID, Studio) VALUES (244, N'ABC Pictures');
INSERT Studio (StudioID, Studio) VALUES (245, N'View Askew Productions');
INSERT Studio (StudioID, Studio) VALUES (246, N'Verti Films');
INSERT Studio (StudioID, Studio) VALUES (247, N'FilmDistrict');
INSERT Studio (StudioID, Studio) VALUES (248, N'Embassy International');
INSERT Studio (StudioID, Studio) VALUES (249, N'Troublemaker Studios');
INSERT Studio (StudioID, Studio) VALUES (250, N'Double Feature Films');
INSERT Studio (StudioID, Studio) VALUES (251, N'Fuqua Films');
INSERT Studio (StudioID, Studio) VALUES (252, N'Artisan Entertainment');
INSERT Studio (StudioID, Studio) VALUES (253, N'Black Bear Pictures');
INSERT Studio (StudioID, Studio) VALUES (255, N'Yellow Bird');
INSERT Studio (StudioID, Studio) VALUES (256, N'Svensk Filmindustri');
INSERT Studio (StudioID, Studio) VALUES (257, N'This Is That');
INSERT Studio (StudioID, Studio) VALUES (258, N'The Weinstein Company');
INSERT Studio (StudioID, Studio) VALUES (259, N'Eli Tossell Film');
INSERT Studio (StudioID, Studio) VALUES (260, N'UK Film Council');
INSERT Studio (StudioID, Studio) VALUES (261, N'Hammer Film Productions');
INSERT Studio (StudioID, Studio) VALUES (262, N'Commonwealth United Entertainment');
INSERT Studio (StudioID, Studio) VALUES (263, N'BHE Films');
INSERT Studio (StudioID, Studio) VALUES (264, N'Cannon Productions');
INSERT Studio (StudioID, Studio) VALUES (265, N'Wise Owl Studios');


INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tom', N'Cruise', CAST(N'1962-07-03T15:06:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sam', N'Neill', CAST(N'1947-09-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Laura', N'Dern', CAST(N'1967-02-10T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jeff', N'Goldblum', CAST(N'1952-10-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Richard', N'Attenborough', CAST(N'1923-08-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Samuel L.', N'Jackson', CAST(N'1948-12-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tobey', N'Maguire', CAST(N'1975-06-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Willem', N'Dafoe', CAST(N'1955-07-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kirsten', N'Dunst', CAST(N'1982-04-30T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Naomi', N'Watts', CAST(N'1968-09-28T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jack', N'Black', CAST(N'1969-08-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Adrien', N'Brody', CAST(N'1973-04-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Andy', N'Serkis', CAST(N'1964-04-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Brandon', N'Routh', CAST(N'1979-10-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kate', N'Bosworth', CAST(N'1983-01-02T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kevin', N'Spacey', CAST(N'1959-07-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Leonardo', N'DiCaprio', CAST(N'1974-11-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kate', N'Winslet', CAST(N'1975-10-05T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Billy', N'Zane', CAST(N'1966-02-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bill', N'Paxton', CAST(N'1955-05-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Steve', N'Carell', CAST(N'1962-08-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Morgan', N'Freeman', CAST(N'1937-06-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kevin', N'Costner', CAST(N'1955-01-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dennis', N'Hopper', CAST(N'1936-05-17T00:00:00.000' AS DateTime), CAST(N'2010-05-29T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ben', N'Affleck', CAST(N'1972-08-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Josh', N'Hartnett', CAST(N'1978-07-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kate', N'Beckinsale', CAST(N'1973-07-26T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Cuba', N'Gooding Jr.', CAST(N'1968-01-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jon', N'Voight', CAST(N'1938-12-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Alec', N'Baldwin', CAST(N'1958-04-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tom', N'Sizemore', CAST(N'1961-11-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dan', N'Aykroyd', CAST(N'1952-07-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Shia', N'LaBeouf', CAST(N'1986-06-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'Turturro', CAST(N'1957-02-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Peter', N'Cullen', CAST(N'1941-07-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Hugo', N'Weaving', CAST(N'1960-04-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Megan', N'Fox', CAST(N'1986-05-16T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Daniel', N'Radcliffe', CAST(N'1989-07-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ralph', N'Fiennes', CAST(N'1962-12-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Brendan', N'Gleeson', CAST(N'1955-03-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Gary', N'Oldman', CAST(N'1958-03-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael', N'Gambon', CAST(N'1940-10-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Alan', N'Rickman', CAST(N'1946-02-21T00:00:00.000' AS DateTime), CAST(N'2016-01-14T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Emma', N'Thompson', CAST(N'1959-04-15T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Helena', N'Bonham Carter', CAST(N'1966-05-26T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Robbie', N'Coltrane', CAST(N'1950-03-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Emma', N'Watson', CAST(N'1990-04-15T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rupert', N'Grint', CAST(N'1988-08-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Robin', N'Wright', CAST(N'1966-04-08T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Anthony', N'Hopkins', CAST(N'1937-12-31T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'Malkovich', CAST(N'1953-12-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ray', N'Winstone', CAST(N'1957-02-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Angelina', N'Jolie', CAST(N'1975-06-04T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'Goodman', CAST(N'1952-06-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jerry', N'Seinfeld', CAST(N'1954-04-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Renee', N'Zellweger', CAST(N'1969-04-25T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Matthew', N'Broderick', CAST(N'1962-03-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Chris', N'Rock', CAST(N'1965-02-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ray', N'Liotta', CAST(N'1954-12-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rip', N'Torn', CAST(N'1931-02-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Johnny', N'Depp', CAST(N'1963-06-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Geoffrey', N'Rush', CAST(N'1951-07-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Orlando', N'Bloom', CAST(N'1977-01-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Keira', N'Knightley', CAST(N'1985-03-26T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bill', N'Nighy', CAST(N'1949-12-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jonathan', N'Pryce', CAST(N'1947-06-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Stellan', N'Skarsgard', CAST(N'1951-06-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Will', N'Smith', CAST(N'1968-09-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Patton', N'Oswalt', CAST(N'1969-01-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ian', N'Holm', CAST(N'1931-09-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Brian', N'Dennehy', CAST(N'1938-07-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Peter', N'O''Toole', CAST(N'1932-08-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Brian', N'Cox', CAST(N'1946-06-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Brad', N'Pitt', CAST(N'1963-12-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Diane', N'Kruger', CAST(N'1976-07-15T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Eric', N'Bana', CAST(N'1968-08-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Christian', N'Bale', CAST(N'1974-01-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael', N'Caine', CAST(N'1933-03-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Liam', N'Neeson', CAST(N'1952-06-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Katie', N'Holmes', CAST(N'1978-12-18T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rutger', N'Hauer', CAST(N'1944-01-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ken', N'Watanabe', CAST(N'1959-10-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'James', N'Fox', CAST(N'1939-05-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Christopher', N'Lee', CAST(N'1922-05-27T00:00:00.000' AS DateTime), CAST(N'2015-06-07T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Pierce', N'Brosnan', CAST(N'1953-05-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Halle', N'Berry', CAST(N'1966-08-14T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rosamund', N'Pike', CAST(N'1979-01-28T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Judi', N'Dench', CAST(N'1934-12-09T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'Cleese', CAST(N'1939-10-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael', N'Madsen', CAST(N'1958-09-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Samantha', N'Bond', CAST(N'1961-11-27T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mel', N'Gibson', CAST(N'1956-01-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Danny', N'Glover', CAST(N'1946-07-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Joe', N'Pesci', CAST(N'1943-02-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rene', N'Russo', CAST(N'1954-02-17T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jet', N'Li', CAST(N'1963-04-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bruce', N'Willis', CAST(N'1955-03-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Billy Bob', N'Thornton', CAST(N'1955-08-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Liv', N'Tyler', CAST(N'1977-07-01T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Steve', N'Buscemi', CAST(N'1957-12-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Owen', N'Wilson', CAST(N'1968-11-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael Clarke', N'Duncan', CAST(N'1957-12-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Peter', N'Stormare', CAST(N'1953-08-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tommy Lee', N'Jones', CAST(N'1946-09-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jamie', N'Foxx', CAST(N'1967-12-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jessica', N'Biel', CAST(N'1982-03-03T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ving', N'Rhames', CAST(N'1959-05-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Donald', N'Sutherland', CAST(N'1935-07-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'James', N'Woods', CAST(N'1947-04-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Colin', N'Farrell', CAST(N'1976-05-31T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ciaran', N'Hinds', CAST(N'1953-02-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sophie', N'Marceau', CAST(N'1966-11-17T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Robert', N'Carlyle', CAST(N'1961-04-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Denise', N'Richards', CAST(N'1971-02-17T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Desmond', N'Llewelyn', CAST(N'1913-09-12T00:00:00.000' AS DateTime), CAST(N'1999-12-19T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Russell', N'Crowe', CAST(N'1964-04-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Paul', N'Bettany', CAST(N'1971-05-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dakota', N'Fanning', CAST(N'1994-02-23T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tim', N'Robbins', CAST(N'1958-10-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Matt', N'Damon', CAST(N'1970-10-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Julia', N'Stiles', CAST(N'1981-03-28T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'David', N'Strathairn', CAST(N'1949-01-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Paddy', N'Considine', CAST(N'1974-09-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Albert', N'Finney', CAST(N'1936-05-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Monica', N'Bellucci', CAST(N'1964-09-30T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Laurence', N'Fishburne', CAST(N'1961-07-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Carrie-Anne', N'Moss', CAST(N'1967-08-21T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Keanu', N'Reeves', CAST(N'1964-09-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tom', N'Hanks', CAST(N'1956-07-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Audrey', N'Tautou', CAST(N'1976-08-09T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ian', N'McKellen', CAST(N'1939-05-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jean', N'Reno', CAST(N'1948-07-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jurgen', N'Prochnow', CAST(N'1941-06-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Richard', N'Harris', CAST(N'1930-10-01T00:00:00.000' AS DateTime), CAST(N'2002-10-25T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'Hurt', CAST(N'1940-01-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dennis', N'Quaid', CAST(N'1954-04-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jake', N'Gyllenhaal', CAST(N'1980-12-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Billy', N'Connolly', CAST(N'1942-11-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Hiroyuki', N'Sanada', CAST(N'1960-10-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Cillian', N'Murphy', CAST(N'1976-05-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michelle', N'Yeoh', CAST(N'1962-08-06T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'George', N'Clooney', CAST(N'1961-05-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mark', N'Wahlberg', CAST(N'1971-06-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John C.', N'Reilly', CAST(N'1965-05-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mary Elizabeth', N'Mastrantonio', CAST(N'1958-11-17T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael', N'Ironside', CAST(N'1950-02-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Philip Seymour', N'Hoffman', CAST(N'1967-07-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michelle', N'Monaghan', CAST(N'1976-03-23T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Simon', N'Pegg', CAST(N'1970-02-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Nicolas', N'Cage', CAST(N'1964-01-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Christian', N'Slater', CAST(N'1969-08-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ewan', N'McGregor', CAST(N'1971-03-31T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Natalie', N'Portman', CAST(N'1981-06-09T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Hayden', N'Christensen', CAST(N'1981-04-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Frank', N'Oz', CAST(N'1944-05-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ian', N'McDiarmid', CAST(N'1944-08-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Temuera', N'Morrison', CAST(N'1960-12-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Billy', N'Crystal', CAST(N'1948-03-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'James', N'Coburn', CAST(N'1928-08-31T00:00:00.000' AS DateTime), CAST(N'2002-11-18T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ray', N'Park', CAST(N'1974-08-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Anthony', N'Daniels', CAST(N'1946-02-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kenny', N'Baker', CAST(N'1934-08-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Peter', N'Mayhew', CAST(N'1944-05-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Joe Don', N'Baker', CAST(N'1936-02-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Elliott', N'Gould', CAST(N'1938-08-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Andy', N'Garcia', CAST(N'1956-04-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Julia', N'Roberts', CAST(N'1967-10-28T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Don', N'Cheadle', CAST(N'1964-11-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Timothy', N'Olyphant', CAST(N'1968-05-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jackie', N'Chan', CAST(N'1954-04-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Steve', N'Coogan', CAST(N'1965-10-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Arnold', N'Schwarzenegger', CAST(N'1947-07-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Maggie', N'Q', CAST(N'1979-05-22T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sammo', N'Hung', CAST(N'1952-01-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jim', N'Broadbent', CAST(N'1949-05-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Luke', N'Wilson', CAST(N'1971-09-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jeremy', N'Irons', CAST(N'1948-09-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Edward', N'Norton', CAST(N'1969-08-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Vince', N'Vaughn', CAST(N'1970-03-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Cate', N'Blanchett', CAST(N'1969-05-14T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jude', N'Law', CAST(N'1972-12-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Brent', N'Spiner', CAST(N'1949-02-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Viggo', N'Mortensen', CAST(N'1958-10-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Elijah', N'Wood', CAST(N'1981-01-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'James', N'Cromwell', CAST(N'1940-01-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bridget', N'Moynahan', CAST(N'1971-04-28T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Daniel', N'Craig', CAST(N'1968-03-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Max', N'von Sydow', CAST(N'1929-04-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Linda', N'Hamilton', CAST(N'1956-09-26T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Edward', N'Furlong', CAST(N'1977-08-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Robert', N'Patrick', CAST(N'1958-11-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kenneth', N'Branagh', CAST(N'1960-12-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Denzel', N'Washington', CAST(N'1954-12-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Djimon', N'Hounsou', CAST(N'1964-04-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jennifer', N'Connelly', CAST(N'1970-12-12T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Arnold', N'Vosloo', CAST(N'1962-06-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Joaquin', N'Phoenix', CAST(N'1974-10-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Oliver', N'Reed', CAST(N'1937-02-13T00:00:00.000' AS DateTime), CAST(N'1999-05-02T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jamie Lee', N'Curtis', CAST(N'1958-11-22T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tom', N'Arnold', CAST(N'1959-03-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Charlton', N'Heston', CAST(N'1924-10-04T00:00:00.000' AS DateTime), CAST(N'2008-04-05T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tia', N'Carrere', CAST(N'1967-01-02T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Art', N'Malik', CAST(N'1952-11-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Eliza', N'Dushku', CAST(N'1980-12-30T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Daniel', N'Day-Lewis', CAST(N'1957-04-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Cameron', N'Diaz', CAST(N'1972-08-30T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Milla', N'Jovovich', CAST(N'1975-12-17T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Chris', N'Tucker', CAST(N'1972-08-31T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sean', N'Bean', CAST(N'1959-04-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Karl', N'Urban', CAST(N'1972-06-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Linda', N'Fiorentino', CAST(N'1958-03-09T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jack', N'Nicholson', CAST(N'1937-04-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Martin', N'Sheen', CAST(N'1940-08-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Paul', N'Giamatti', CAST(N'1967-06-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'F. Murray', N'Abraham', CAST(N'1939-10-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Charles', N'Dance', CAST(N'1946-10-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Catherine', N'Zeta-Jones', CAST(N'1969-09-25T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Adam', N'Sandler', CAST(N'1966-09-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Christopher', N'Walken', CAST(N'1943-03-31T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'David', N'Hasselhoff', CAST(N'1952-07-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Henry', N'Winkler', CAST(N'1945-10-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jim', N'Carrey', CAST(N'1962-01-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jennifer', N'Aniston', CAST(N'1969-02-11T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Emmanuelle', N'Beart', CAST(N'1963-08-14T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Robert', N'Duvall', CAST(N'1931-01-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tea', N'Leoni', CAST(N'1966-02-25T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jon', N'Favreau', CAST(N'1966-10-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Vanessa', N'Redgrave', CAST(N'1937-01-30T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Paul', N'Newman', CAST(N'1925-01-26T00:00:00.000' AS DateTime), CAST(N'2008-09-26T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Stanley', N'Tucci', CAST(N'1960-11-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ed', N'Harris', CAST(N'1950-11-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Christopher', N'Plummer', CAST(N'1929-12-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Clive', N'Owen', CAST(N'1964-10-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bill', N'Pullman', CAST(N'1953-12-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mary', N'McDonnell', CAST(N'1952-04-28T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Randy', N'Quaid', CAST(N'1950-10-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Hugh', N'Jackman', CAST(N'1968-10-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Patrick', N'Stewart', CAST(N'1940-07-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Famke', N'Janssen', CAST(N'1965-11-05T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sean', N'Connery', CAST(N'1930-08-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael', N'Biehn', CAST(N'1956-07-31T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rachel', N'Weisz', CAST(N'1971-03-07T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Will', N'Ferrell', CAST(N'1967-07-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sacha', N'Baron Cohen', CAST(N'1971-10-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dan', N'Castellaneta', CAST(N'1957-05-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Julie', N'Kavner', CAST(N'1950-09-07T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Nancy', N'Cartwright', CAST(N'1957-10-25T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Yeardley', N'Smith', CAST(N'1964-07-03T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Patrick', N'McGoohan', CAST(N'1928-03-19T00:00:00.000' AS DateTime), CAST(N'2009-01-13T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'William', N'Hurt', CAST(N'1950-03-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sigourney', N'Weaver', CAST(N'1949-10-08T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mike', N'Myers', CAST(N'1963-05-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Eddie', N'Murphy', CAST(N'1961-04-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Antonio', N'Banderas', CAST(N'1960-08-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jonathan', N'Frakes', CAST(N'1952-08-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'LeVar', N'Burton', CAST(N'1957-02-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael', N'Dorn', CAST(N'1952-12-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Gates', N'McFadden', CAST(N'1949-03-02T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Marina', N'Sirtis', CAST(N'1955-03-29T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bob', N'Hoskins', CAST(N'1942-10-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Christopher', N'Lloyd', CAST(N'1938-10-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kathleen', N'Turner', CAST(N'1954-06-19T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Christina', N'Ricci', CAST(N'1980-02-12T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Vin', N'Diesel', CAST(N'1967-07-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'William', N'Sadler', CAST(N'1950-04-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bonnie', N'Bedelia', CAST(N'1948-03-25T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Geena', N'Davis', CAST(N'1956-01-21T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kevin', N'Bacon', CAST(N'1958-07-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Gary', N'Sinise', CAST(N'1955-03-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sharon', N'Stone', CAST(N'1958-03-10T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sylvester', N'Stallone', CAST(N'1946-07-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'Lithgow', CAST(N'1945-10-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael', N'Rooker', CAST(N'1955-04-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Harrison', N'Ford', CAST(N'1942-07-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Scarlett', N'Johansson', CAST(N'1984-11-22T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Hilary', N'Swank', CAST(N'1974-07-30T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kathleen', N'Quinlan', CAST(N'1954-11-19T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Joely', N'Richardson', CAST(N'1965-01-09T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sean', N'Pertwee', CAST(N'1964-06-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ron', N'Perlman', CAST(N'1950-04-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Selma', N'Blair', CAST(N'1972-06-23T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ben', N'Stiller', CAST(N'1965-11-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Juliette', N'Lewis', CAST(N'1973-06-21T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Fred', N'Williamson', CAST(N'1938-03-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Snoop', N'Dogg', CAST(N'1971-10-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Chris', N'Penn', CAST(N'1965-10-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Gerard', N'Butler', CAST(N'1969-11-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dominic', N'West', CAST(N'1969-10-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'David', N'Wenham', CAST(N'1965-09-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Lena', N'Headey', CAST(N'1973-10-03T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Malcolm', N'McDowell', CAST(N'1943-06-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'James', N'Doohan', CAST(N'1920-03-03T00:00:00.000' AS DateTime), CAST(N'2005-07-20T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Walter', N'Koenig', CAST(N'1936-09-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'William', N'Shatner', CAST(N'1931-03-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Leonard', N'Nimoy', CAST(N'1931-03-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'DeForest', N'Kelley', CAST(N'1920-01-20T00:00:00.000' AS DateTime), CAST(N'1999-06-11T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'George', N'Takei', CAST(N'1937-04-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Nichelle', N'Nichols', CAST(N'1932-12-28T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ricardo', N'Montalban', CAST(N'1920-11-25T00:00:00.000' AS DateTime), CAST(N'2009-01-14T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Marlon', N'Brando', CAST(N'1924-04-03T00:00:00.000' AS DateTime), CAST(N'2004-07-01T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Gene', N'Hackman', CAST(N'1930-01-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Christopher', N'Reeve', CAST(N'1952-09-25T00:00:00.000' AS DateTime), CAST(N'2004-10-10T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Margot', N'Kidder', CAST(N'1948-10-17T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'James', N'Gandolfini', CAST(N'1961-09-18T00:00:00.000' AS DateTime), CAST(N'2013-06-19T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Val', N'Kilmer', CAST(N'1959-12-31T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jim', N'Caviezel', CAST(N'1968-09-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Uma', N'Thurman', CAST(N'1970-04-29T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Lucy', N'Liu', CAST(N'1968-12-02T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Daryl', N'Hannah', CAST(N'1960-12-03T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'David', N'Carradine', CAST(N'1936-12-08T00:00:00.000' AS DateTime), CAST(N'2009-06-03T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Charles S.', N'Dutton', CAST(N'1951-01-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Pete', N'Postlethwaite', CAST(N'1946-02-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Wesley', N'Snipes', CAST(N'1962-07-31T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kris', N'Kristofferson', CAST(N'1936-06-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Donnie', N'Yen', CAST(N'1963-07-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ryan', N'Phillippe', CAST(N'1974-09-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Robert', N'De Niro', CAST(N'1943-08-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'Rhys-Davies', CAST(N'1944-05-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Stephen', N'Dorff', CAST(N'1973-07-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'James Earl', N'Jones', CAST(N'1931-01-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Julianne', N'Moore', CAST(N'1960-12-03T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Joe', N'Pantoliano', CAST(N'1951-09-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Madeleine', N'Stowe', CAST(N'1958-08-18T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Wes', N'Studi', CAST(N'1947-12-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rudy', N'Youngblood', CAST(N'1982-09-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dalia', N'Hernandez', CAST(N'1985-08-14T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rachel', N'McAdams', CAST(N'1976-10-07T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Isla', N'Fisher', CAST(N'1976-02-03T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jane', N'Seymour', CAST(N'1951-02-15T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael J.', N'Fox', CAST(N'1961-06-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Thomas F.', N'Wilson', CAST(N'1959-04-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Timothy', N'Dalton', CAST(N'1944-03-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Benicio', N'Del Toro', CAST(N'1967-02-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Robert', N'Brown', CAST(N'1921-07-23T00:00:00.000' AS DateTime), CAST(N'2003-11-11T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'David', N'Hedison', CAST(N'1927-05-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'James', N'Franco', CAST(N'1978-04-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'J. K.', N'Simmons', CAST(N'1955-01-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'James', N'Marsden', CAST(N'1973-09-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Frank', N'Langella', CAST(N'1938-01-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kathy', N'Bates', CAST(N'1948-06-28T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bernard', N'Hill', CAST(N'1944-12-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'David', N'Warner', CAST(N'1941-07-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jonah', N'Hill', CAST(N'1983-12-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kim', N'Coates', CAST(N'1959-01-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jennifer', N'Garner', CAST(N'1972-04-17T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael', N'Shannon', CAST(N'1974-08-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mako', N'Iwamatsu', CAST(N'1933-12-10T00:00:00.000' AS DateTime), CAST(N'2006-07-21T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Cary-Hiroyuki', N'Tagawa', CAST(N'1950-09-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'William', N'Fichtner', CAST(N'1956-11-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tomas', N'Arana', CAST(N'1955-04-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Josh', N'Duhamel', CAST(N'1972-11-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tyrese', N'Gibson', CAST(N'1978-12-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bernie', N'Mac', CAST(N'1957-10-05T00:00:00.000' AS DateTime), CAST(N'2008-08-09T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Richard', N'Griffiths', CAST(N'1947-07-31T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jessica', N'Stevenson', CAST(N'1972-11-15T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Robert', N'Pattinson', CAST(N'1986-05-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mark', N'Williams', CAST(N'1959-08-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'David', N'Thewlis', CAST(N'1963-03-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Maggie', N'Smith', CAST(N'1934-12-28T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Julie', N'Walters', CAST(N'1950-02-22T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jason', N'Isaacs', CAST(N'1963-06-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Imelda', N'Staunton', CAST(N'1956-01-09T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tom', N'Felton', CAST(N'1987-09-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Warwick', N'Davis', CAST(N'1970-02-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Timothy', N'Spall', CAST(N'1957-02-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Miguel', N'Sandoval', CAST(N'1951-11-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Joe', N'Manganiello', CAST(N'1976-12-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bruce', N'Campbell', CAST(N'1958-06-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Elizabeth', N'Banks', CAST(N'1974-02-10T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Lucy', N'Lawless', CAST(N'1968-03-29T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Thomas', N'Kretschmann', CAST(N'1962-09-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Colin', N'Hanks', CAST(N'1977-11-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jamie', N'Bell', CAST(N'1986-03-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ioan', N'Gruffudd', CAST(N'1973-10-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jenette', N'Goldstein', CAST(N'1960-02-04T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Glenn', N'Morshower', CAST(N'1959-04-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Patrick', N'Warburton', CAST(N'1964-11-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Larry', N'King', CAST(N'1933-11-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sting', NULL, CAST(N'1951-10-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Oprah', N'Winfrey', CAST(N'1954-01-29T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tress', N'MacNeille', CAST(N'1951-06-20T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'Di Maggio', CAST(N'1968-09-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mackenzie', N'Crook', CAST(N'1971-09-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Naomie', N'Harris', CAST(N'1976-09-06T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Yun-fat', N'Chow', CAST(N'1955-05-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Keith', N'Richards', CAST(N'1943-12-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Omid', N'Djalili', CAST(N'1965-09-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Janeane', N'Garofalo', CAST(N'1964-09-28T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Will', N'Arnett', CAST(N'1970-05-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Julian', N'Glover', CAST(N'1935-03-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Julie', N'Christie', CAST(N'1941-04-14T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'James', N'Cosmo', CAST(N'1948-05-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rose', N'Byrne', CAST(N'1979-07-24T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'David', N'Tennant', CAST(N'1971-04-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Miranda', N'Richardson', CAST(N'1958-03-03T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tom', N'Wilkinson', CAST(N'1948-02-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Missi', N'Pyle', CAST(N'1972-11-16T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Toby', N'Stephens', CAST(N'1969-04-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rick', N'Yune', CAST(N'1971-08-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Colin', N'Salmon', CAST(N'1962-12-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Will', N'Patton', CAST(N'1954-06-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Keith', N'David', CAST(N'1956-06-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Johnny', N'Knoxville', CAST(N'1971-03-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rosario', N'Dawson', CAST(N'1979-05-09T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tony', N'Shalhoub', CAST(N'1953-10-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael', N'Jackson', CAST(N'1958-08-29T00:00:00.000' AS DateTime), CAST(N'2009-06-25T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bryce Dallas', N'Howard', CAST(N'1981-03-02T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Alfred', N'Molina', CAST(N'1953-05-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Daniel Dae', N'Kim', CAST(N'1968-08-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Goldie', NULL, CAST(N'1965-09-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Serena', N'Scott Thomas', CAST(N'1961-09-21T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'David', N'Threlfall', CAST(N'1953-10-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Billy', N'Boyd', CAST(N'1968-08-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Miranda', N'Otto', CAST(N'1967-12-16T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Scott', N'Glenn', CAST(N'1941-01-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Joan', N'Allen', CAST(N'1956-08-20T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Daniel', N'Bruhl', CAST(N'1978-06-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Pam', N'Ferris', CAST(N'1948-05-11T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Lenny', N'Henry', CAST(N'1958-08-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dawn', N'French', CAST(N'1957-10-11T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Paul', N'Whitehouse', CAST(N'1958-05-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Harold', N'Perrineau', CAST(N'1963-08-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jada', N'Pinkett Smith', CAST(N'1971-09-18T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Gina', N'Torres', CAST(N'1969-04-25T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Verne', N'Troyer', CAST(N'1969-01-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Zoe', N'Wanamaker', CAST(N'1949-05-13T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Zoe', N'Saldana', CAST(N'1978-06-19T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Karen', N'Allen', CAST(N'1951-10-05T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Billy', N'Crudup', CAST(N'1968-07-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jonathan Rhys', N'Meyers', CAST(N'1977-07-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Eddie', N'Marsan', CAST(N'1968-06-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Aaron', N'Paul', CAST(N'1979-08-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dougray', N'Scott', CAST(N'1965-11-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Thandie', N'Newton', CAST(N'1972-11-06T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dominic', N'Purcell', CAST(N'1970-02-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mark', N'Ruffalo', CAST(N'1967-11-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jimmy', N'Smits', CAST(N'1955-07-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Terence', N'Stamp', CAST(N'1938-07-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Brian', N'Blessed', CAST(N'1936-10-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Peter', N'Serafinowicz', CAST(N'1972-07-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Teri', N'Hatcher', CAST(N'1964-12-08T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Geoffrey', N'Palmer', CAST(N'1927-06-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Casey', N'Affleck', CAST(N'1975-08-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Lennox', N'Lewis', CAST(N'1965-09-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Wladimir', N'Klitschko', CAST(N'1976-03-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Justin', N'Long', CAST(N'1978-06-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kevin', N'Smith', CAST(N'1970-08-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tim', N'Russ', CAST(N'1956-06-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rob', N'Schneider', CAST(N'1963-10-31T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael', N'Sheen', CAST(N'1969-02-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Philip', N'Glenister', CAST(N'1963-02-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kevin', N'McKidd', CAST(N'1973-09-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Nikolaj', N'Coster-Waldau', CAST(N'1970-07-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Marton', N'Csokas', CAST(N'1966-06-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Alexander', N'Siddig', CAST(N'1965-11-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Eva', N'Green', CAST(N'1980-07-05T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Iain', N'Glen', CAST(N'1961-06-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jane', N'Lynch', CAST(N'1960-07-14T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mario', N'Van Peebles', CAST(N'1957-01-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ron', N'Silver', CAST(N'1946-07-02T00:00:00.000' AS DateTime), CAST(N'2009-03-15T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jeffrey', N'Wright', CAST(N'1965-12-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sean', N'Astin', CAST(N'1971-02-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dominic', N'Monaghan', CAST(N'1976-12-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Alan', N'Tudyk', CAST(N'1971-03-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mads', N'Mikkelsen', CAST(N'1965-11-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Samantha', N'Morton', CAST(N'1977-05-13T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Xander', N'Berkeley', CAST(N'1955-12-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Josh', N'Brolin', CAST(N'1968-02-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Carla', N'Gugino', CAST(N'1971-08-29T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Idris', N'Elba', CAST(N'1972-09-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Chiwetel', N'Ejiofor', CAST(N'1977-07-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Norman', N'Reedus', CAST(N'1969-01-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jimi', N'Mistry', CAST(N'1973-01-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Derek', N'Jacobi', CAST(N'1938-10-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ralf', N'Moller', CAST(N'1959-01-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'David', N'Hemmings', CAST(N'1941-11-18T00:00:00.000' AS DateTime), CAST(N'2003-12-03T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tommy', N'Flanagan', CAST(N'1965-07-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Stephen', N'Graham', CAST(N'1973-08-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tom', N'Hardy', CAST(N'1977-09-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Lee', N'Evans', CAST(N'1964-02-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Luke', N'Perry', CAST(N'1966-10-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Brad', N'Dourif', CAST(N'1950-03-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tzi', N'Ma', CAST(N'1962-06-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ziyi', N'Zhang', CAST(N'1979-02-09T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Vincent', N'D''Onofrio', CAST(N'1959-06-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Graham', N'Greene', CAST(N'1952-06-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Anthony', N'Quinn', CAST(N'1915-04-21T00:00:00.000' AS DateTime), CAST(N'2001-06-03T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Helen', N'Hunt', CAST(N'1963-06-15T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Eddie', N'Izzard', CAST(N'1962-02-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jeroen', N'Krabbe', CAST(N'1944-12-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kristin', N'Scott Thomas', CAST(N'1960-05-24T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Emilio', N'Estevez', CAST(N'1962-05-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Anthony', N'LaPaglia', CAST(N'1959-01-31T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Charlie', N'Hunnam', CAST(N'1980-04-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael', N'Lonsdale', CAST(N'1931-05-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Robert', N'Loggia', CAST(N'1930-01-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Adam', N'Baldwin', CAST(N'1962-02-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Harry', N'Connick Jr.', CAST(N'1967-09-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Frank', N'Welker', CAST(N'1946-03-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rebecca', N'Romijn', CAST(N'1972-11-06T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Anna', N'Paquin', CAST(N'1982-07-24T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Shawn', N'Ashmore', CAST(N'1979-10-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Alan', N'Cumming', CAST(N'1965-01-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John C.', N'McGinley', CAST(N'1959-08-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tilda', N'Swinton', CAST(N'1960-11-05T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Gavin', N'Rossdale', CAST(N'1965-10-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Amy', N'Adams', CAST(N'1974-08-20T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mos', N'Def', CAST(N'1973-12-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Hank', N'Azaria', CAST(N'1964-04-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Harry', N'Shearer', CAST(N'1943-12-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Pamela', N'Hayden', CAST(N'1953-11-28T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Joe', N'Mantegna', CAST(N'1947-11-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dennis', N'Haysbert', CAST(N'1954-06-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'M. Night', N'Shyamalan', CAST(N'1970-08-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jesse', N'Eisenberg', CAST(N'1983-10-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Justin', N'Timberlake', CAST(N'1981-01-31T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rupert', N'Everett', CAST(N'1959-05-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Eric', N'Idle', CAST(N'1943-03-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Julie', N'Andrews', CAST(N'1935-10-01T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Amy', N'Poehler', CAST(N'1971-09-16T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ian', N'McShane', CAST(N'1942-09-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Seth', N'Rogen', CAST(N'1982-04-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jennifer', N'Saunders', CAST(N'1958-07-06T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Joan', N'Rivers', CAST(N'1933-06-08T00:00:00.000' AS DateTime), CAST(N'2014-09-04T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tom', N'Morello', CAST(N'1964-05-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mel', N'Blanc', CAST(N'1908-05-30T00:00:00.000' AS DateTime), CAST(N'1989-07-10T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jeffrey', N'Jones', CAST(N'1946-09-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Danny', N'Trejo', CAST(N'1944-05-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dennis', N'Franz', CAST(N'1944-10-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Colm', N'Meaney', CAST(N'1953-05-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'Leguizamo', CAST(N'1964-07-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Max', N'Grodenchik', CAST(N'1952-11-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Giovanni', N'Ribisi', CAST(N'1974-12-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ted', N'Danson', CAST(N'1947-12-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dennis', N'Farina', CAST(N'1944-02-29T00:00:00.000' AS DateTime), CAST(N'2013-07-22T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bryan', N'Cranston', CAST(N'1956-03-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Nathan', N'Fillion', CAST(N'1971-03-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Meat', N'Loaf', CAST(N'1947-09-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jared', N'Leto', CAST(N'1971-12-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Marc', N'Alaimo', CAST(N'1942-05-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Craig', N'Fairbrass', CAST(N'1964-01-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Joaquim', N'de Almeida', CAST(N'1957-03-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Anne', N'Archer', CAST(N'1947-08-24T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Izabella', N'Scorupco', CAST(N'1970-06-04T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Minnie', N'Driver', CAST(N'1970-01-31T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Aaron', N'Eckhart', CAST(N'1968-03-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Adewale', N'Akinnuoye-Agbaje', CAST(N'1967-08-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Walton', N'Goggins', CAST(N'1971-11-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rupert', N'Evans', CAST(N'1977-03-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jeffrey', N'Tambor', CAST(N'1944-07-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'David Hyde', N'Pierce', CAST(N'1959-04-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jason', N'Bateman', CAST(N'1969-01-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Amy', N'Smart', CAST(N'1976-03-26T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Carmen', N'Electra', CAST(N'1972-04-20T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Terry', N'Crews', CAST(N'1968-07-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Paul Michael', N'Glaser', CAST(N'1943-03-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'David', N'Soul', CAST(N'1943-08-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael', N'Fassbender', CAST(N'1977-04-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Peter', N'Mensah', CAST(N'1959-08-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kate', N'Mulgrew', CAST(N'1955-04-29T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Alan', N'Dale', CAST(N'1947-05-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Wil', N'Wheaton', CAST(N'1972-07-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Majel', N'Barrett', CAST(N'1932-02-23T00:00:00.000' AS DateTime), CAST(N'2008-12-18T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Whoopi', N'Goldberg', CAST(N'1955-11-13T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ned', N'Beatty', CAST(N'1937-07-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Glenn', N'Ford', CAST(N'1916-05-01T00:00:00.000' AS DateTime), CAST(N'2006-08-30T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jason', N'Robards', CAST(N'1922-07-26T00:00:00.000' AS DateTime), CAST(N'2000-12-26T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mickey', N'Rourke', CAST(N'1952-09-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Lisa', N'Bonet', CAST(N'1967-11-16T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Gabriel', N'Byrne', CAST(N'1950-05-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Seth', N'Green', CAST(N'1974-02-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sonny', N'Chiba', CAST(N'1939-01-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Gordon', N'Liu', CAST(N'1955-08-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Paul', N'McGann', CAST(N'1959-11-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Lance', N'Henriksen', CAST(N'1940-05-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Biao', N'Yuen', CAST(N'1957-07-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Luke', N'Goss', CAST(N'1968-09-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Danny', N'John-Jules', CAST(N'1960-09-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Paul', N'Walker', CAST(N'1973-09-12T00:00:00.000' AS DateTime), CAST(N'2013-11-30T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kevin', N'Pollak', CAST(N'1957-10-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mike', N'McShane', CAST(N'1955-06-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael', N'Wincott', CAST(N'1958-01-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Aidan', N'Gillen', CAST(N'1968-04-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Denholm', N'Elliott', CAST(N'1922-05-31T00:00:00.000' AS DateTime), CAST(N'1992-10-06T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Alexei', N'Sayle', CAST(N'1952-08-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Alice', N'Krige', CAST(N'1954-06-28T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dwight', N'Schultz', CAST(N'1947-11-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Robert', N'Picardo', CAST(N'1953-10-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Donal', N'Logue', CAST(N'1966-02-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Patrick', N'Bergin', CAST(N'1951-02-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Andreas', N'Katsulas', CAST(N'1946-05-18T00:00:00.000' AS DateTime), CAST(N'2006-02-13T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'David', N'Bowie', CAST(N'1947-01-08T00:00:00.000' AS DateTime), CAST(N'2016-01-10T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bradley', N'Cooper', CAST(N'1975-01-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rebecca', N'De Mornay', CAST(N'1959-08-29T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Elisabeth', N'Shue', CAST(N'1963-10-06T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Flea', NULL, CAST(N'1962-10-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Walter', N'Gotell', CAST(N'1924-03-15T00:00:00.000' AS DateTime), CAST(N'1997-05-05T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jessica', N'Alba', CAST(N'1981-04-28T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Devon', N'Aoki', CAST(N'1982-08-10T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Powers', N'Boothe', CAST(N'1948-06-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Brittany', N'Murphy', CAST(N'1977-11-10T00:00:00.000' AS DateTime), CAST(N'2009-12-20T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Morena', N'Baccarin', CAST(N'1979-06-02T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Summer', N'Glau', CAST(N'1981-07-24T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kim', N'Basinger', CAST(N'1953-12-08T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Edward', N'Fox', CAST(N'1937-04-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rowan', N'Atkinson', CAST(N'1955-01-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mark', N'Lenard', CAST(N'1924-10-15T00:00:00.000' AS DateTime), CAST(N'1996-11-22T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mark', N'Hamill', CAST(N'1951-09-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Carrie', N'Fisher', CAST(N'1956-10-21T00:00:00.000' AS DateTime), CAST(N'2016-12-27T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Billy Dee', N'Williams', CAST(N'1937-04-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'David', N'Prowse', CAST(N'1935-07-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Alec', N'Guinness', CAST(N'1914-04-02T00:00:00.000' AS DateTime), CAST(N'2000-08-05T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Roger', N'Moore', CAST(N'1927-10-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Richard', N'Kiel', CAST(N'1939-09-13T00:00:00.000' AS DateTime), CAST(N'2014-09-10T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bernard', N'Lee', CAST(N'1908-01-10T00:00:00.000' AS DateTime), CAST(N'1981-01-16T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Lois', N'Maxwell', CAST(N'1927-02-14T00:00:00.000' AS DateTime), CAST(N'2007-09-29T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'Travolta', CAST(N'1954-02-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Danny', N'DeVito', CAST(N'1944-11-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bette', N'Midler', CAST(N'1945-12-01T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Delroy', N'Lindo', CAST(N'1952-11-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Harvey', N'Keitel', CAST(N'1939-05-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jeremy', N'Renner', CAST(N'1971-01-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sam', N'Rockwell', CAST(N'1968-11-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Zooey', N'Deschanel', CAST(N'1980-01-17T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Clint', N'Eastwood', CAST(N'1930-05-31T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mike', N'Colter', CAST(N'1976-08-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Anthony', N'Mackie', CAST(N'1978-09-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael', N'Pena', CAST(N'1976-01-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Gwyneth', N'Paltrow', CAST(N'1972-09-27T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'R. Lee', N'Ermey', CAST(N'1944-03-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Richard', N'Roundtree', CAST(N'1942-07-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tim', N'Curry', CAST(N'1946-04-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bill', N'Murray', CAST(N'1950-09-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Harold', N'Ramis', CAST(N'1944-11-21T00:00:00.000' AS DateTime), CAST(N'2014-02-24T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ernie', N'Hudson', CAST(N'1945-12-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rick', N'Moranis', CAST(N'1953-04-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sean', N'Young', CAST(N'1959-11-20T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Edward James', N'Olmos', CAST(N'1947-02-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'James', N'Hong', CAST(N'1929-02-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Al', N'Leong', CAST(N'1952-09-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kim', N'Cattrall', CAST(N'1956-08-21T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Iman', NULL, CAST(N'1955-07-25T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rene', N'Auberjonois', CAST(N'1940-06-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Yaphet', N'Kotto', CAST(N'1939-11-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jesse', N'Ventura', CAST(N'1951-07-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Annette', N'Bening', CAST(N'1958-05-29T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Cathy', N'Moriarty', CAST(N'1960-11-29T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Holly', N'Hunter', CAST(N'1958-03-20T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Lorraine', N'Bracco', CAST(N'1954-10-02T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Paul', N'Sorvino', CAST(N'1939-04-13T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael', N'Imperioli', CAST(N'1966-03-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ben', N'Kingsley', CAST(N'1943-12-31T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Christina', N'Applegate', CAST(N'1971-11-25T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Paul', N'Rudd', CAST(N'1969-04-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Al', N'Pacino', CAST(N'1940-04-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michelle', N'Pfeiffer', CAST(N'1958-04-29T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jeff', N'Bridges', CAST(N'1949-12-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Charles', N'Grodin', CAST(N'1935-04-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jessica', N'Lange', CAST(N'1949-04-20T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'Gielgud', CAST(N'1904-04-14T00:00:00.000' AS DateTime), CAST(N'2000-05-21T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Om', N'Puri', CAST(N'1950-10-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Emily', N'Watson', CAST(N'1967-01-14T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Taye', N'Diggs', CAST(N'1971-01-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Quentin', N'Tarantino', CAST(N'1963-03-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Salma', N'Hayek', CAST(N'1966-09-02T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Cheech', N'Marin', CAST(N'1946-07-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'Saxon', CAST(N'1935-08-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rodney A.', N'Grant', CAST(N'1959-03-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Pamela', N'Anderson', CAST(N'1967-07-01T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Paul', N'Reiser', CAST(N'1957-03-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tony', N'Leung', CAST(N'1962-06-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Maggie', N'Cheung', CAST(N'1964-09-20T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Nick', N'Frost', CAST(N'1972-03-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kevin', N'Eldon', CAST(N'1960-10-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Olivia', N'Colman', CAST(N'1974-01-30T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bill', N'Bailey', CAST(N'1964-01-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Edward', N'Woodward', CAST(N'1930-06-01T00:00:00.000' AS DateTime), CAST(N'2009-11-16T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Adam', N'Buxton', CAST(N'1969-06-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Stephen', N'Merchant', CAST(N'1974-11-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Martin', N'Freeman', CAST(N'1971-09-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Danny', N'Aiello', CAST(N'1933-06-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Javier', N'Bardem', CAST(N'1969-03-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jason', N'Statham', CAST(N'1967-07-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mena', N'Suvari', CAST(N'1979-02-13T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Allison', N'Janney', CAST(N'1959-11-19T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Scott', N'Bakula', CAST(N'1954-10-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Gary', N'Busey', CAST(N'1944-06-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Patsy', N'Kensit', CAST(N'1968-03-04T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Heath', N'Ledger', CAST(N'1979-04-04T00:00:00.000' AS DateTime), CAST(N'2008-01-22T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Anne', N'Hathaway', CAST(N'1982-11-12T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kate', N'Mara', CAST(N'1983-02-27T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sam', N'Elliott', CAST(N'1944-08-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Nick', N'Nolte', CAST(N'1941-02-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Lou', N'Ferrigno', CAST(N'1951-11-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Stan', N'Lee', CAST(N'1922-12-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Hugh', N'Grant', CAST(N'1960-09-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Skeet', N'Ulrich', CAST(N'1970-01-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jewel', NULL, CAST(N'1974-05-23T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Heather', N'Graham', CAST(N'1970-01-29T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Christopher', N'Eccleston', CAST(N'1964-02-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'David', N'Schneider', CAST(N'1963-05-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jonny Lee', N'Miller', CAST(N'1972-11-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kelly', N'Macdonald', CAST(N'1976-02-23T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Keith', N'Allen', CAST(N'1953-09-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bruno', N'Ganz', CAST(N'1941-03-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kirstie', N'Alley', CAST(N'1951-01-12T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Andy', N'Lau', CAST(N'1961-09-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'David', N'Niven', CAST(N'1910-03-01T00:00:00.000' AS DateTime), CAST(N'1983-07-29T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Peter', N'Sellers', CAST(N'1925-09-08T00:00:00.000' AS DateTime), CAST(N'1980-07-24T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ursula', N'Andress', CAST(N'1936-03-19T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Woody', N'Allen', CAST(N'1935-12-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bernard', N'Cribbins', CAST(N'1928-12-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ronnie', N'Corbett', CAST(N'1930-12-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Burt', N'Kwouk', CAST(N'1930-07-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'Le Mesurier', CAST(N'1912-04-05T00:00:00.000' AS DateTime), CAST(N'1983-11-15T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Stirling', N'Moss', CAST(N'1929-09-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Peter', N'Cushing', CAST(N'1913-05-26T00:00:00.000' AS DateTime), CAST(N'1994-08-11T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Drew', N'Barrymore', CAST(N'1975-02-22T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Erika', N'Eleniak', CAST(N'1969-09-29T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Donald', N'Pleasence', CAST(N'1919-10-05T00:00:00.000' AS DateTime), CAST(N'1995-02-02T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tom', N'Skerritt', CAST(N'1933-08-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Harry Dean', N'Stanton', CAST(N'1926-07-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'George', N'Lazenby', CAST(N'1939-09-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Diana', N'Rigg', CAST(N'1938-07-20T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Telly', N'Savalas', CAST(N'1922-01-21T00:00:00.000' AS DateTime), CAST(N'1994-01-22T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Joanna', N'Lumley', CAST(N'1946-05-01T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tim', N'Roth', CAST(N'1961-05-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Eric', N'Stoltz', CAST(N'1961-09-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rosanna', N'Arquette', CAST(N'1959-08-10T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Frances', N'McDormand', CAST(N'1957-06-23T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'William H.', N'Macy', CAST(N'1950-03-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Britt', N'Ekland', CAST(N'1942-10-06T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Herve', N'Villechaize', CAST(N'1943-04-23T00:00:00.000' AS DateTime), CAST(N'1993-09-04T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Robert', N'Downey Jr.', CAST(N'1965-04-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jeff', N'Daniels', CAST(N'1955-02-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tate', N'Donovan', CAST(N'1963-09-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Shirley', N'MacLaine', CAST(N'1934-04-24T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Noel', N'Coward', CAST(N'1899-12-16T00:00:00.000' AS DateTime), CAST(N'1973-03-26T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Marlene', N'Dietrich', CAST(N'1901-12-27T00:00:00.000' AS DateTime), CAST(N'1992-05-06T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Frank', N'Sinatra', CAST(N'1915-12-12T00:00:00.000' AS DateTime), CAST(N'1998-05-14T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Buster', N'Keaton', CAST(N'1895-10-04T00:00:00.000' AS DateTime), CAST(N'1966-02-01T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Stephen', N'Baldwin', CAST(N'1966-05-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Chazz', N'Palminteri', CAST(N'1952-05-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Guy', N'Pearce', CAST(N'1967-10-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Stephen', N'Tobolowsky', CAST(N'1951-05-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dylan', N'Moran', CAST(N'1971-11-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Reece', N'Shearsmith', CAST(N'1969-08-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tamsin', N'Greig', CAST(N'1966-07-12T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Matt', N'Lucas', CAST(N'1974-03-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Andie', N'MacDowell', CAST(N'1958-04-21T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Simon', N'Callow', CAST(N'1949-06-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'Hannah', CAST(N'1962-04-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Honor', N'Blackman', CAST(N'1925-08-22T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Gert', N'Frobe', CAST(N'1913-02-25T00:00:00.000' AS DateTime), CAST(N'1988-09-05T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Toshiro', N'Mifune', CAST(N'1920-04-01T00:00:00.000' AS DateTime), CAST(N'1997-12-24T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Takashi', N'Shimura', CAST(N'1905-03-12T00:00:00.000' AS DateTime), CAST(N'1982-02-11T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Morgan', N'Spurlock', CAST(N'1970-11-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Chris', N'Evans', CAST(N'1981-06-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mark', N'Strong', CAST(N'1963-08-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kelsey', N'Grammer', CAST(N'1955-02-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'DiMaggio', CAST(N'1968-09-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Benedict', N'Cumberbatch', CAST(N'1976-07-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Evangeline', N'Lilly', CAST(N'1979-08-03T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'James', N'Nesbitt', CAST(N'1965-01-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sylvester', N'McCoy', CAST(N'1943-08-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Stephen', N'Fry', CAST(N'1957-08-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Chris', N'Pratt', CAST(N'1979-06-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dave', N'Bautista', CAST(N'1969-01-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Karen', N'Gillan', CAST(N'1987-11-28T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Glenn', N'Close', CAST(N'1947-03-19T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Vivienne', N'Jolie-Pitt', CAST(N'2008-07-12T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'James', N'McAvoy', CAST(N'1979-04-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jennifer', N'Lawrence', CAST(N'1990-08-15T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Nicholas', N'Hoult', CAST(N'1989-12-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ellen', N'Page', CAST(N'1987-02-21T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Peter', N'Dinklage', CAST(N'1969-06-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sebastian', N'Stan', CAST(N'1982-08-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Cobie', N'Smulders', CAST(N'1982-04-03T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Frank', N'Grillo', CAST(N'1965-06-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Emily', N'VanCamp', CAST(N'1986-05-12T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Hayley', N'Atwell', CAST(N'1982-04-05T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Robert', N'Redford', CAST(N'1936-08-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Toby', N'Jones', CAST(N'1966-09-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Garry', N'Shandling', CAST(N'1949-11-29T00:00:00.000' AS DateTime), CAST(N'2016-03-24T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Georges', N'St-Pierre', CAST(N'1981-05-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Josh', N'Hutcherson', CAST(N'1992-10-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Liam', N'Hemsworth', CAST(N'1990-01-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Woody', N'Harrelson', CAST(N'1961-07-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Natalie', N'Dormer', CAST(N'1982-02-11T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Andrew', N'Garfield', CAST(N'1983-08-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Emma', N'Stone', CAST(N'1988-11-06T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sally', N'Field', CAST(N'1946-11-06T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Denis', N'Leary', CAST(N'1957-08-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Matthew', N'McConaughey', CAST(N'1969-11-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jessica', N'Chastain', CAST(N'1977-03-24T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kristen', N'Bell', CAST(N'1980-07-18T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kristen', N'Wiig', CAST(N'1973-08-22T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Miranda', N'Cosgrove', CAST(N'1993-05-14T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Russell', N'Brand', CAST(N'1975-06-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ken', N'Jeong', CAST(N'1969-07-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Lenny', N'Kravitz', CAST(N'1964-05-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dwayne', N'Johnson', CAST(N'1972-05-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michelle', N'Rodriguez', CAST(N'1978-07-12T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jordana', N'Brewster', CAST(N'1980-04-26T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ludacris', NULL, CAST(N'1977-09-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Gal', N'Gadot', CAST(N'1985-04-30T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Helen', N'Mirren', CAST(N'1945-07-26T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bill', N'Hader', CAST(N'1978-06-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sandra', N'Bullock', CAST(N'1964-07-26T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Henry', N'Cavill', CAST(N'1983-05-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Chris', N'Hemsworth', CAST(N'1983-08-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tom', N'Hiddleston', CAST(N'1981-02-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kat', N'Dennings', CAST(N'1986-06-13T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ray', N'Stevenson', CAST(N'1964-05-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Chris', N'O''Dowd', CAST(N'1979-10-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Donald', N'Glover', CAST(N'1983-09-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Gwendoline', N'Christie', CAST(N'1978-10-28T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sean', N'Harris', NULL, NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mindy', N'Kaling', CAST(N'1979-06-24T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kyle', N'MacLachlan', CAST(N'1959-02-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rashida', N'Jones', CAST(N'1976-02-25T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Christoph', N'Waltz', CAST(N'1956-10-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jon', N'Hamm', CAST(N'1971-03-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael', N'Keaton', CAST(N'1951-09-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Aaron', N'Taylor-Johnson', CAST(N'1990-06-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Elizabeth', N'Olsen', CAST(N'1989-02-16T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'James', N'Spader', CAST(N'1960-02-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tony', N'Jaa', CAST(N'1976-02-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ronda', N'Rousey', CAST(N'1987-02-01T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kurt', N'Russell', CAST(N'1951-03-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Adam', N'Driver', CAST(N'1983-11-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Daisy', N'Ridley', CAST(N'1992-04-10T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'Boyega', CAST(N'1992-03-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Oscar', N'Isaac', CAST(N'1979-03-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Lupita', N'Nyong''o', CAST(N'1983-03-01T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Clark', N'Gregg', CAST(N'1962-04-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Marion', N'Cotillard', CAST(N'1975-09-30T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Joseph', N'Gordon-Levitt', CAST(N'1981-02-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Matthew', N'Modine', CAST(N'1959-03-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ray', N'Romano', CAST(N'1957-12-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Seann William', N'Scott', CAST(N'1976-10-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Josh', N'Peck', CAST(N'1986-11-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jennifer', N'Lopez', CAST(N'1969-07-24T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Queen', N'Latifah', CAST(N'1970-03-18T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Aziz', N'Ansari', CAST(N'1983-02-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Nicki', N'Minaj', CAST(N'1982-12-08T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kunal', N'Nayyar', CAST(N'1981-04-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rebel', N'Wilson', CAST(N'1980-03-02T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kristen', N'Stewart', CAST(N'1990-04-09T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Taylor', N'Lautner', CAST(N'1992-02-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kellan', N'Lutz', CAST(N'1985-03-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Maggie', N'Grace', CAST(N'1983-09-21T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rhys', N'Ifans', CAST(N'1967-07-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'David', N'Schwimmer', CAST(N'1966-11-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Martin', N'Short', CAST(N'1950-03-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Vinnie', N'Jones', CAST(N'1965-01-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Alice', N'Eve', CAST(N'1982-02-06T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Nicole', N'Scherzinger', CAST(N'1978-06-29T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Nick', N'Moran', CAST(N'1969-12-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Patrick', N'Dempsey', CAST(N'1966-01-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Penelope', N'Cruz', CAST(N'1974-04-28T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dustin', N'Hoffman', CAST(N'1937-08-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Danny', N'McBride', CAST(N'1976-12-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jean-Claude', N'Van Damme', CAST(N'1960-10-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Eva', N'Mendes', CAST(N'1974-03-05T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Zach', N'Galifianakis', CAST(N'1969-10-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Neil Patrick', N'Harris', CAST(N'1973-06-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Anton', N'Yelchin', CAST(N'1989-03-11T00:00:00.000' AS DateTime), CAST(N'2016-06-19T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Katy', N'Perry', CAST(N'1984-10-25T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tim', N'Allen', CAST(N'1953-06-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Joan', N'Cusack', CAST(N'1962-10-11T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Barbara', N'Windsor', CAST(N'1937-08-06T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tom', N'Berenger', CAST(N'1949-05-31T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jason', N'Segel', CAST(N'1980-01-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'America', N'Ferrera', CAST(N'1984-04-18T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Christopher', N'Mintz-Plasse', CAST(N'1989-06-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ashley', N'Jensen', CAST(N'1969-08-11T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sam', N'Worthington', CAST(N'1976-08-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael', N'York', CAST(N'1942-03-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'Cusack', CAST(N'1966-06-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Oliver', N'Platt', CAST(N'1960-01-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'George', N'Segal', CAST(N'1934-02-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ed', N'Asner', CAST(N'1929-11-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mike', N'Tyson', CAST(N'1966-06-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Maggie', N'Gyllenhaal', CAST(N'1977-11-16T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Eric', N'Roberts', CAST(N'1956-04-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Charlize', N'Theron', CAST(N'1975-08-07T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Meryl', N'Streep', CAST(N'1949-06-22T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Amanda', N'Seyfried', CAST(N'1985-12-03T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Colin', N'Firth', CAST(N'1960-09-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dominic', N'Cooper', CAST(N'1978-06-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Olga', N'Kurylenko', CAST(N'1979-11-14T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Gemma', N'Arterton', CAST(N'1986-02-02T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Terrence', N'Howard', CAST(N'1969-03-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dick', N'Van Dyke', CAST(N'1925-12-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mickey', N'Rooney', CAST(N'1920-09-23T00:00:00.000' AS DateTime), CAST(N'2014-04-06T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ricky', N'Gervais', CAST(N'1961-06-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Robin', N'Williams', CAST(N'1951-07-21T00:00:00.000' AS DateTime), CAST(N'2014-08-11T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'George', N'Carlin', CAST(N'1937-05-12T00:00:00.000' AS DateTime), CAST(N'2008-06-22T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Nicole', N'Kidman', CAST(N'1967-06-20T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jason', N'Lee', CAST(N'1970-04-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Barbra', N'Streisand', CAST(N'1942-04-24T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ellen', N'DeGeneres', CAST(N'1958-01-26T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Claire', N'Danes', CAST(N'1979-04-12T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Martin', N'Lawrence', CAST(N'1965-04-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Chadwick', N'Boseman', CAST(N'1976-11-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tom', N'Holland', CAST(N'1996-06-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Diane', N'Keaton', CAST(N'1946-01-05T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Eugene', N'Levy', CAST(N'1946-12-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kate', N'McKinnon', CAST(N'1984-01-06T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jeffrey Dean', N'Morgan', CAST(N'1966-04-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jason', N'Momoa', CAST(N'1979-08-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kevin', N'Hart', CAST(N'1979-07-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dana', N'Carvey', CAST(N'1955-06-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ryan', N'Reynolds', CAST(N'1976-10-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Margot', N'Robbie', CAST(N'1990-07-02T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jai', N'Courtney', CAST(N'1986-03-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Cara', N'Delevingne', CAST(N'1992-08-12T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sophie', N'Turner', CAST(N'1969-02-21T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Alice', N'Braga', CAST(N'1983-04-15T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Brendan', N'Fraser', CAST(N'1968-12-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rosie', N'O''Donnell', CAST(N'1962-03-21T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tim', N'McInnerny', CAST(N'1956-09-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Robert', N'Wagner', CAST(N'1930-02-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rob', N'Lowe', CAST(N'1964-03-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Elizabeth', N'Hurley', CAST(N'1965-06-10T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Matt', N'Dillon', CAST(N'1964-02-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sarah', N'Silverman', CAST(N'1970-12-01T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Julia', N'Louis-Dreyfus', CAST(N'1961-01-13T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Hayden', N'Panettiere', CAST(N'1989-08-21T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Roddy', N'McDowall', CAST(N'1928-09-17T00:00:00.000' AS DateTime), CAST(N'1998-10-03T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ming-Na', N'Wen', CAST(N'1963-11-20T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Pat', N'Morita', CAST(N'1932-06-28T00:00:00.000' AS DateTime), CAST(N'2005-11-24T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Joseph', N'Fiennes', CAST(N'1970-05-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Cary', N'Elwes', CAST(N'1962-10-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Demi', N'Moore', CAST(N'1962-11-11T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kevin', N'Kline', CAST(N'1947-10-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Hugh', N'Laurie', CAST(N'1959-06-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Lindsay', N'Lohan', CAST(N'1986-07-02T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Liev', N'Schreiber', CAST(N'1967-10-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Donnie', N'Wahlberg', CAST(N'1969-08-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Beau', N'Bridges', CAST(N'1941-12-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'James', N'Caan', CAST(N'1940-03-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Chris', N'O''Donnell', CAST(N'1970-06-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Elizabeth', N'Taylor', CAST(N'1932-02-27T00:00:00.000' AS DateTime), CAST(N'2011-03-23T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Meg', N'Ryan', CAST(N'1961-11-19T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Whitney', N'Houston', CAST(N'1963-08-09T00:00:00.000' AS DateTime), CAST(N'2012-02-11T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Macaulay', N'Culkin', CAST(N'1980-08-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Daniel', N'Stern', CAST(N'1957-08-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael', N'Douglas', CAST(N'1944-09-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kiefer', N'Sutherland', CAST(N'1966-12-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Winona', N'Ryder', CAST(N'1971-10-29T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Richard E.', N'Grant', CAST(N'1957-05-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jodie', N'Foster', CAST(N'1962-11-19T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sissy', N'Spacek', CAST(N'1949-12-25T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jack', N'Lemmon', CAST(N'1925-02-08T00:00:00.000' AS DateTime), CAST(N'2001-06-27T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Walter', N'Matthau', CAST(N'1920-10-01T00:00:00.000' AS DateTime), CAST(N'2000-07-01T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'Candy', CAST(N'1950-10-31T00:00:00.000' AS DateTime), CAST(N'1994-03-04T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Anjelica', N'Huston', CAST(N'1951-07-08T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Raul', N'Julia', CAST(N'1940-03-09T00:00:00.000' AS DateTime), CAST(N'1994-10-24T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Robert', N'Mitchum', CAST(N'1917-08-06T00:00:00.000' AS DateTime), CAST(N'1997-07-01T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Gregory', N'Peck', CAST(N'1916-04-05T00:00:00.000' AS DateTime), CAST(N'2003-06-12T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Charlie', N'Sheen', CAST(N'1965-09-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Lloyd', N'Bridges', CAST(N'1913-01-15T00:00:00.000' AS DateTime), CAST(N'1998-03-10T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kristy', N'Swanson', CAST(N'1969-12-19T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jack', N'Palance', CAST(N'1919-02-18T00:00:00.000' AS DateTime), CAST(N'2006-11-13T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Patrick', N'Swayze', CAST(N'1952-08-18T00:00:00.000' AS DateTime), CAST(N'2009-09-14T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Richard', N'Gere', CAST(N'1949-08-31T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Corey', N'Feldman', CAST(N'1971-07-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ethan', N'Hawke', CAST(N'1970-11-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Arsenio', N'Hall', CAST(N'1955-02-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jon', N'Lovitz', CAST(N'1957-07-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Paul', N'Hogan', CAST(N'1939-10-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Linda', N'Kozlowski', CAST(N'1958-01-07T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Luis', N'Guzman', CAST(N'1956-08-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Leslie', N'Nielsen', CAST(N'1926-02-11T00:00:00.000' AS DateTime), CAST(N'2010-11-28T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bryan', N'Brown', CAST(N'1947-06-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tom', N'Selleck', CAST(N'1945-01-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Steve', N'Guttenberg', CAST(N'1958-08-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Brigitte', N'Nielsen', CAST(N'1963-07-15T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dean', N'Stockwell', CAST(N'1936-03-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Forest', N'Whitaker', CAST(N'1961-07-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Cher', NULL, CAST(N'1946-05-20T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Richard', N'Dreyfuss', CAST(N'1947-10-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Susan', N'Sarandon', CAST(N'1946-10-04T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kelly', N'McGillis', CAST(N'1957-07-09T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Anthony', N'Edwards', CAST(N'1962-07-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ralph', N'Macchio', CAST(N'1961-11-04T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Victor', N'Wong', CAST(N'1927-07-30T00:00:00.000' AS DateTime), CAST(N'2001-09-12T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jennifer', N'Grey', CAST(N'1960-03-26T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Burt', N'Young', CAST(N'1940-04-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Carl', N'Weathers', CAST(N'1948-01-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dolph', N'Lundgren', CAST(N'1957-11-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Don', N'Ameche', CAST(N'1908-05-31T00:00:00.000' AS DateTime), CAST(N'1993-12-06T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jessica', N'Tandy', CAST(N'1909-06-07T00:00:00.000' AS DateTime), CAST(N'1994-09-11T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Chevy', N'Chase', CAST(N'1943-10-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Terry', N'Gilliam', CAST(N'1940-11-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Judge', N'Reinhold', CAST(N'1957-05-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Phoebe', N'Cates', CAST(N'1963-07-16T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'William', N'Zabka', CAST(N'1965-10-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sarah Jessica', N'Parker', CAST(N'1965-03-25T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jim', N'Belushi', CAST(N'1954-06-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sondra', N'Locke', CAST(N'1944-05-28T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Louis', N'Gossett Jr.', CAST(N'1936-05-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mr.', N'T', CAST(N'1952-05-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dudley', N'Moore', CAST(N'1935-04-19T00:00:00.000' AS DateTime), CAST(N'2002-03-27T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Liza', N'Minnelli', CAST(N'1946-03-12T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Burt', N'Reynolds', CAST(N'1936-02-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dom', N'DeLuise', CAST(N'1933-08-01T00:00:00.000' AS DateTime), CAST(N'2009-05-04T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Farrah', N'Fawcett', CAST(N'1947-02-02T00:00:00.000' AS DateTime), CAST(N'2009-06-25T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dean', N'Martin', CAST(N'1917-06-07T00:00:00.000' AS DateTime), CAST(N'1995-12-25T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sammy', N'Davis Jr.', CAST(N'1925-12-08T00:00:00.000' AS DateTime), CAST(N'1990-05-16T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Peter', N'Fonda', CAST(N'1940-02-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Nigel', N'Havers', CAST(N'1951-11-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael', N'Palin', CAST(N'1943-05-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jane', N'Fonda', CAST(N'1937-12-21T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Lily', N'Tomlin', CAST(N'1939-09-01T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dolly', N'Parton', CAST(N'1946-01-19T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Gene', N'Wilder', CAST(N'1933-06-11T00:00:00.000' AS DateTime), CAST(N'2016-08-29T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Richard', N'Pryor', CAST(N'1940-12-01T00:00:00.000' AS DateTime), CAST(N'2005-12-10T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kareem', N'Abdul-Jabbar', CAST(N'1947-04-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ethel', N'Merman', CAST(N'1908-01-16T00:00:00.000' AS DateTime), CAST(N'1984-02-15T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Goldie', N'Hawn', CAST(N'1945-11-21T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'Belushi', CAST(N'1949-01-24T00:00:00.000' AS DateTime), CAST(N'1982-03-05T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bo', N'Derek', CAST(N'1956-11-20T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sam J.', N'Jones', CAST(N'1954-08-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Steve', N'Martin', CAST(N'1945-08-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jim', N'Henson', CAST(N'1936-09-24T00:00:00.000' AS DateTime), CAST(N'1990-05-16T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Olivia', N'Newton-John', CAST(N'1948-09-26T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Roy', N'Scheider', CAST(N'1932-11-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jackie', N'Gleason', CAST(N'1916-02-26T00:00:00.000' AS DateTime), CAST(N'1987-06-24T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dirk', N'Bogarde', CAST(N'1921-03-28T00:00:00.000' AS DateTime), CAST(N'1999-05-08T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Laurence', N'Olivier', CAST(N'1907-05-22T00:00:00.000' AS DateTime), CAST(N'1989-07-11T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Robert', N'Shaw', CAST(N'1927-08-09T00:00:00.000' AS DateTime), CAST(N'1978-08-28T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Eli', N'Wallach', CAST(N'1915-12-07T00:00:00.000' AS DateTime), CAST(N'2014-06-24T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Henry', N'Fonda', CAST(N'1905-05-16T00:00:00.000' AS DateTime), CAST(N'1982-08-12T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Richard', N'O''Brien', CAST(N'1942-03-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Warren', N'Beatty', CAST(N'1937-03-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mel', N'Brooks', CAST(N'1926-06-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Steve', N'McQueen', CAST(N'1930-03-24T00:00:00.000' AS DateTime), CAST(N'1980-11-07T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Faye', N'Dunaway', CAST(N'1941-01-14T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Fred', N'Astaire', CAST(N'1899-05-10T00:00:00.000' AS DateTime), CAST(N'1987-06-22T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Robert', N'Vaughn', CAST(N'1932-11-22T00:00:00.000' AS DateTime), CAST(N'2016-11-11T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Lorne', N'Greene', CAST(N'1915-02-12T00:00:00.000' AS DateTime), CAST(N'1987-09-11T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Talia', N'Shire', CAST(N'1946-04-25T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'George', N'Kennedy', CAST(N'1925-02-18T00:00:00.000' AS DateTime), CAST(N'2016-02-28T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Linda', N'Blair', CAST(N'1959-01-22T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ron', N'Howard', CAST(N'1954-03-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Peter', N'Ustinov', CAST(N'1921-04-16T00:00:00.000' AS DateTime), CAST(N'2004-03-28T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Terry', N'Thomas', CAST(N'1911-07-10T00:00:00.000' AS DateTime), CAST(N'1990-01-08T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ernest', N'Borgnine', CAST(N'1917-01-24T00:00:00.000' AS DateTime), CAST(N'2012-07-08T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Topol', NULL, CAST(N'1935-09-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Angela', N'Lansbury', CAST(N'1925-10-16T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ali', N'MacGraw', CAST(N'1939-04-01T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Burt', N'Lancaster', CAST(N'1913-11-02T00:00:00.000' AS DateTime), CAST(N'1994-10-20T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'George C.', N'Scott', CAST(N'1927-10-18T00:00:00.000' AS DateTime), CAST(N'1999-09-22T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bridget', N'Fonda', CAST(N'1964-01-27T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Lee', N'Marvin', CAST(N'1924-02-19T00:00:00.000' AS DateTime), CAST(N'1987-08-29T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'Wayne', CAST(N'1907-05-26T00:00:00.000' AS DateTime), CAST(N'1979-06-11T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Wilford', N'Brimley', CAST(N'1934-09-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tony', N'Curtis', CAST(N'1925-06-03T00:00:00.000' AS DateTime), CAST(N'2010-09-29T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Anne', N'Bancroft', CAST(N'1931-09-17T00:00:00.000' AS DateTime), CAST(N'2005-06-06T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Charles', N'Bronson', CAST(N'1921-11-03T00:00:00.000' AS DateTime), CAST(N'2003-08-30T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Lee', N'Van Cleef', CAST(N'1925-01-09T00:00:00.000' AS DateTime), CAST(N'1989-12-16T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'James', N'Garner', CAST(N'1928-04-07T00:00:00.000' AS DateTime), CAST(N'2014-07-19T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Eva Marie', N'Saint', CAST(N'1924-07-04T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Omar', N'Sharif', CAST(N'1932-04-10T00:00:00.000' AS DateTime), CAST(N'2015-07-10T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Audrey', N'Hepburn', CAST(N'1929-05-04T00:00:00.000' AS DateTime), CAST(N'1993-01-20T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rex', N'Harrison', CAST(N'1908-03-05T00:00:00.000' AS DateTime), CAST(N'1990-06-02T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'Lennon', CAST(N'1940-10-09T00:00:00.000' AS DateTime), CAST(N'1980-12-08T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Paul', N'McCartney', CAST(N'1942-06-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'George', N'Harrison', CAST(N'1943-02-25T00:00:00.000' AS DateTime), CAST(N'2001-11-29T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ringo', N'Starr', CAST(N'1940-07-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Richard', N'Burton', CAST(N'1925-11-10T00:00:00.000' AS DateTime), CAST(N'1984-08-05T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'George', N'Peppard', CAST(N'1928-10-01T00:00:00.000' AS DateTime), CAST(N'1994-05-08T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'James', N'Stewart', CAST(N'1908-05-20T00:00:00.000' AS DateTime), CAST(N'1997-07-02T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Richard', N'Widmark', CAST(N'1914-12-26T00:00:00.000' AS DateTime), CAST(N'2008-03-24T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Spencer', N'Tracy', CAST(N'1900-04-05T00:00:00.000' AS DateTime), CAST(N'1967-06-10T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Alfred', N'Hitchcock', CAST(N'1899-08-13T00:00:00.000' AS DateTime), CAST(N'1980-04-29T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Curt', N'Jurgens', CAST(N'1915-12-13T00:00:00.000' AS DateTime), CAST(N'1982-06-18T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sophia', N'Loren', CAST(N'1934-09-20T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kirk', N'Douglas', CAST(N'1916-12-09T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Janet', N'Leigh', CAST(N'1927-07-06T00:00:00.000' AS DateTime), CAST(N'2004-10-03T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'Mills', CAST(N'1908-02-22T00:00:00.000' AS DateTime), CAST(N'2005-04-23T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Angie', N'Dickinson', CAST(N'1931-09-30T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jack', N'Hawkins', CAST(N'1910-09-14T00:00:00.000' AS DateTime), CAST(N'1973-07-18T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Cary', N'Grant', CAST(N'1904-01-18T00:00:00.000' AS DateTime), CAST(N'1986-11-29T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'James', N'Mason', CAST(N'1909-05-15T00:00:00.000' AS DateTime), CAST(N'1984-07-27T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Marilyn', N'Monroe', CAST(N'1926-06-01T00:00:00.000' AS DateTime), CAST(N'1962-08-05T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rock', N'Hudson', CAST(N'1925-11-17T00:00:00.000' AS DateTime), CAST(N'1985-10-02T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Yul', N'Brynner', CAST(N'1920-07-11T00:00:00.000' AS DateTime), CAST(N'1985-10-10T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Vincent', N'Price', CAST(N'1911-05-27T00:00:00.000' AS DateTime), CAST(N'1993-10-25T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Deborah', N'Kerr', CAST(N'1921-09-30T00:00:00.000' AS DateTime), CAST(N'2007-10-16T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bing', N'Crosby', CAST(N'1903-05-03T00:00:00.000' AS DateTime), CAST(N'1977-10-14T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Grace', N'Kelly', CAST(N'1929-11-12T00:00:00.000' AS DateTime), CAST(N'1982-09-14T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jane', N'Russell', CAST(N'1921-06-21T00:00:00.000' AS DateTime), CAST(N'2011-02-28T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Gene', N'Kelly', CAST(N'1912-08-23T00:00:00.000' AS DateTime), CAST(N'1996-02-02T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Zsa Zsa', N'Gabor', CAST(N'1917-02-06T00:00:00.000' AS DateTime), CAST(N'2016-12-18T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Vivien', N'Leigh', CAST(N'1913-11-05T00:00:00.000' AS DateTime), CAST(N'1967-07-08T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Humphrey', N'Bogart', CAST(N'1899-12-25T00:00:00.000' AS DateTime), CAST(N'1957-01-14T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Katharine', N'Hepburn', CAST(N'1907-05-12T00:00:00.000' AS DateTime), CAST(N'2003-06-29T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Gary', N'Cooper', CAST(N'1901-05-07T00:00:00.000' AS DateTime), CAST(N'1961-05-13T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ingrid', N'Bergman', CAST(N'1915-08-29T00:00:00.000' AS DateTime), CAST(N'1982-08-29T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Errol', N'Flynn', CAST(N'1909-06-20T00:00:00.000' AS DateTime), CAST(N'1959-10-14T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Olivia', N'de Havilland', CAST(N'1916-07-01T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Basil', N'Rathbone', CAST(N'1892-06-13T00:00:00.000' AS DateTime), CAST(N'1967-07-21T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Charlie', N'Chaplin', CAST(N'1889-04-16T00:00:00.000' AS DateTime), CAST(N'1977-12-25T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Clark', N'Gable', CAST(N'1901-02-01T00:00:00.000' AS DateTime), CAST(N'1960-11-16T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Shelley', N'Winters', CAST(N'1920-08-18T00:00:00.000' AS DateTime), CAST(N'2006-01-14T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Judy', N'Garland', CAST(N'1922-06-10T00:00:00.000' AS DateTime), CAST(N'1969-06-22T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'James', N'Cagney', CAST(N'1899-07-17T00:00:00.000' AS DateTime), CAST(N'1986-03-30T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Orson', N'Welles', CAST(N'1915-05-06T00:00:00.000' AS DateTime), CAST(N'1985-10-10T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Greta', N'Garbo', CAST(N'1905-09-18T00:00:00.000' AS DateTime), CAST(N'1990-04-15T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Boris', N'Karloff', CAST(N'1887-11-23T00:00:00.000' AS DateTime), CAST(N'1969-02-02T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bela', N'Lugosi', CAST(N'1882-10-20T00:00:00.000' AS DateTime), CAST(N'1956-08-16T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bruce', N'Lee', CAST(N'1940-11-27T00:00:00.000' AS DateTime), CAST(N'1973-07-20T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'James', N'Tien', CAST(N'1942-05-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Chuck', N'Norris', CAST(N'1940-03-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Robert', N'Wall', CAST(N'1939-08-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dan', N'Inosanto', CAST(N'1936-07-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jim', N'Kelly', CAST(N'1946-05-05T00:00:00.000' AS DateTime), CAST(N'2013-06-29T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bolo', N'Yeung', CAST(N'1946-07-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Graham', N'Chapman', CAST(N'1941-01-08T00:00:00.000' AS DateTime), CAST(N'1989-10-04T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Terry', N'Jones', CAST(N'1942-02-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Benny', N'Urquidez', CAST(N'1952-06-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'DMX', NULL, CAST(N'1970-12-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Liam', N'Cunningham', CAST(N'1961-06-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dave', N'Franco', CAST(N'1985-06-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Takeshi', N'Kaneshiro', CAST(N'1973-10-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Iko', N'Uwais', CAST(N'1983-02-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jason Scott', N'Lee', CAST(N'1966-11-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Takeshi', N'Kitano', CAST(N'1947-01-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tadanobu', N'Asano', CAST(N'1973-11-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Kate', N'Hudson', CAST(N'1979-04-19T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ethan', N'Suplee', CAST(N'1976-05-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sean', N'Penn', CAST(N'1960-08-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Drea', N'de Matteo', CAST(N'1972-01-19T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Stephen', N'Chow', CAST(N'1962-06-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ricky', N'Tomlinson', CAST(N'1939-09-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sienna', N'Miller', CAST(N'1981-12-28T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Luke', N'Evans', CAST(N'1979-04-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'James', N'Purefoy', CAST(N'1964-06-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Brie', N'Larson', CAST(N'1989-10-01T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sharlto', N'Copley', CAST(N'1973-11-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Armie', N'Hammer', CAST(N'1986-08-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Dev', N'Patel', CAST(N'1990-04-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rufus', N'Sewell', CAST(N'1967-10-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Vincent', N'Cassel', CAST(N'1966-11-23T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Gerard', N'Depardieu', CAST(N'1948-12-27T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Emilia', N'Clarke', CAST(N'1986-10-23T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Matt', N'Smith', CAST(N'1982-10-28T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Curtis "50 Cent"', N'Jackson', CAST(N'1975-07-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jaimie', N'Alexander', CAST(N'1984-03-12T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Alicia', N'Silverstone', CAST(N'1976-10-04T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Elle', N'Macpherson', CAST(N'1964-03-29T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Peter', N'Boyle', CAST(N'1935-10-18T00:00:00.000' AS DateTime), CAST(N'2006-12-12T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Sven-Ole', N'Thorsen', CAST(N'1944-09-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Bill', N'Duke', CAST(N'1943-02-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Grace', N'Jones', CAST(N'1948-05-19T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Andre the', N'Giant', CAST(N'1946-05-19T00:00:00.000' AS DateTime), CAST(N'1993-01-27T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Rose', N'McGowan', CAST(N'1973-09-05T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Amber', N'Heard', CAST(N'1986-04-22T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Reese', N'Witherspoon', CAST(N'1976-03-22T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Chris', N'Pine', CAST(N'1980-08-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Matthew', N'Fox', CAST(N'1966-07-14T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'January', N'Jones', CAST(N'1978-01-05T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Matt', N'LeBlanc', CAST(N'1967-07-25T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Taron', N'Egerton', CAST(N'1989-11-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Chloe Grace', N'Moretz', CAST(N'1997-02-10T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Charlie', N'Cox', CAST(N'1982-12-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Tobias', N'Menzies', CAST(N'1974-03-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Steven', N'Seagal', CAST(N'1952-04-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Nigel', N'Hawthorne', CAST(N'1929-04-05T00:00:00.000' AS DateTime), CAST(N'2001-12-26T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Haley Joel', N'Osment', CAST(N'1988-04-10T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jack', N'O''Connell', CAST(N'1990-08-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Domhnall', N'Gleeson', CAST(N'1983-05-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Carey', N'Mulligan', CAST(N'1985-05-28T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Shelley', N'Duvall', CAST(N'1949-07-07T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Lauren', N'Bacall', CAST(N'1924-09-16T00:00:00.000' AS DateTime), CAST(N'2014-08-12T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mandy', N'Patinkin', CAST(N'1952-11-30T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Stacy', N'Keach', CAST(N'1941-06-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Avery', N'Brooks', CAST(N'1948-10-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Maximilian', N'Schell', CAST(N'1930-12-08T00:00:00.000' AS DateTime), CAST(N'2014-02-01T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'David', N'McCallum', CAST(N'1933-09-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Stanley', N'Baker', CAST(N'1928-02-28T00:00:00.000' AS DateTime), CAST(N'1976-06-28T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Zachary', N'Quinto', CAST(N'1977-06-02T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'John', N'Cho', CAST(N'1972-06-16T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Peter', N'Weller', CAST(N'1947-06-24T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jason', N'Mewes', CAST(N'1974-06-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Shannen', N'Doherty', CAST(N'1971-04-12T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Joey Lauren', N'Adams', CAST(N'1968-01-09T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ryan', N'Gosling', CAST(N'1980-11-12T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Christina', N'Hendricks', CAST(N'1975-05-03T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jon', N'Bernthal', CAST(N'1976-09-20T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Emily', N'Mortimer', CAST(N'1971-12-01T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Melissa', N'McCarthy', CAST(N'1970-08-26T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Leslie', N'Jones', CAST(N'1967-09-07T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael', N'Cera', CAST(N'1988-06-07T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mary Elizabeth', N'Winstead', CAST(N'1984-11-28T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Pam', N'Grier', CAST(N'1949-05-26T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Robert', N'Forster', CAST(N'1941-07-13T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Jennifer Jason', N'Leigh', CAST(N'1962-02-05T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Channing', N'Tatum', CAST(N'1980-04-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Alex', N'Winter', CAST(N'1965-07-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mira', N'Sorvino', CAST(N'1967-09-28T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Elias', N'Koteas', CAST(N'1961-03-11T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ashley', N'Judd', CAST(N'1968-04-19T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'David', N'Oyelowo', CAST(N'1976-04-01T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Eddie', N'Redmayne', CAST(N'1982-01-06T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Felicity', N'Jones', CAST(N'1983-10-17T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Diego', N'Luna', CAST(N'1979-12-29T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ben', N'Mendelsohn', CAST(N'1969-04-03T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Alicia', N'Vikander', CAST(N'1988-10-03T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Juliette', N'Binoche', CAST(N'1964-03-09T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Clancy', N'Brown', CAST(N'1959-01-05T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Greg', N'Kinnear', CAST(N'1963-06-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Noomi', N'Rapace', CAST(N'1979-12-28T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Michael', N'Nyqvist', CAST(N'1960-11-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Seth', N'MacFarlane', CAST(N'1973-10-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mila', N'Kunis', CAST(N'1983-08-14T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Paul', N'Dano', CAST(N'1984-06-19T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Mary Lynn', N'Rajskub', CAST(N'1971-06-22T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Thomas', N'Jane', CAST(N'1969-02-22T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Emily', N'Blunt', CAST(N'1983-02-23T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Keith', N'Carradine', CAST(N'1949-08-08T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Lou Diamond', N'Phillips', CAST(N'1962-02-17T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Corey', N'Haim', CAST(N'1971-12-23T00:00:00.000' AS DateTime), CAST(N'2010-03-10T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'William', N'Baldwin', CAST(N'1963-02-21T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Ice', N'Cube', CAST(N'1969-06-15T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Anna', N'Faris', CAST(N'1976-11-29T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Andy', N'Samberg', CAST(N'1978-08-18T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Nick', N'Offerman', CAST(N'1970-06-26T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Gillian', N'Anderson', CAST(N'1968-08-09T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Raquel', N'Welch', CAST(N'1940-09-05T00:00:00.000' AS DateTime), NULL, N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Richard', N'Chamberlain', CAST(N'1934-03-31T00:00:00.000' AS DateTime), NULL, N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'James', N'Dean', CAST(N'1931-02-08T00:00:00.000' AS DateTime), CAST(N'1955-09-30T00:00:00.000' AS DateTime), N'Male');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Natalie', N'Wood', CAST(N'1938-07-20T00:00:00.000' AS DateTime), CAST(N'1981-11-29T00:00:00.000' AS DateTime), N'Female');
INSERT Actor ( FirstName, FamilyName, DoB, DoD, Gender) VALUES( N'Hedy', N'Lamarr', CAST(N'1914-11-09T00:00:00.000' AS DateTime), CAST(N'2000-01-19T00:00:00.000' AS DateTime), N'Female');

INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1, N'Jurassic Park', CAST(N'1993-06-11T00:00:00.000' AS DateTime), 4, 41, N'People clone dinosaurs. Dinosaurs promptly eat people. Hilarity ensues! Wait, not hiliarity, the other one, horror! Yes, that''s it, horror.', 241, 1, 8, 126, 2, 63000000, 1029939903, 3, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (2, N'Spider-Man', CAST(N'2002-05-03T00:00:00.000' AS DateTime), 11, 53, N'High-school nerd Peter Parker is bitten by a genetically engineered spider leading to him developing super powers and transforming into Spider-Man! Cue much dreary teenage angst and some nonsense about great responsi-blah-bity. Shut up already and punch some bad guys you weed.', 241, 1, 4, 121, 4, 140000000, 821708551, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (3, N'King Kong', CAST(N'2005-12-14T00:00:00.000' AS DateTime), 12, 42, N'Dizzy blonde actress falls for drippy writer on an adventure with an ambitious film director. Oh, and something about a big monkey. Sorry, ape.', 167, 1, 8, 187, 4, 207000000, 550500000, 4, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (5, N'Superman Returns', CAST(N'2006-07-14T00:00:00.000' AS DateTime), 14, 43, N'The Man of Steel returns from a sabbatical to discover that nobody gives a rat''s a**. Except for Lois Lane who apparently cared about him so much that she ran off and had a kid with some other dude. Seriously, Bryan Singer gave up X-Men for this drivel. What was he thinking?!', 241, 1, 4, 154, 4, 204000000, 391081192, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (6, N'Titanic', CAST(N'1998-01-23T00:00:00.000' AS DateTime), 15, 4, N'Bleurgh! Sorry, that''s the sound of me vomiting. Coincidentally it''s also my opinion of this abysmal, cynical, money-making machine. If you enjoy this film you are a horrible human being and should be ashamed of yourself.', 241, 1, 6, 194, 3, 200000000, 2186772302, 14, 11);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (7, N'Evan Almighty', CAST(N'2007-06-22T00:00:00.000' AS DateTime), 16, 44, N'Morgan Freeman is God. Steve Carell tries to be funny. There''s also a giant boat made of wood. Depressingly, the boat upstages most of the cast.', 241, 1, 7, 96, 2, 175000000, 173418781, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (8, N'Waterworld', CAST(N'1995-07-28T00:00:00.000' AS DateTime), 17, 45, N'Kevin Costner grows a pair of gills! This instantly makes him 100% more interesting than any of the other parts he has ever played. Unfortunately this still puts him at zero on the interesting scale. Dennis Hopper also hams it up as The Baddie. You can tell he''s a baddie ''cos he''s only got one eye.', 241, 1, 8, 135, 3, 175000000, 264218220, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (9, N'Pearl Harbor', CAST(N'2001-05-21T00:00:00.000' AS DateTime), 18, 7, N'NEEEEEOWWM!!! BOOM!!! DAKKA DAKKA DAKKA!!! Is how the original script written by Michael Bay read. Fortunately some other, sane, people added some actual dialogue to the script. Unfortunately, most of this dialogue is delivered by Ben Affleck and Josh Hartnett. First he giveth, then he taketh away.', 241, 1, 5, 183, 3, 140000000, 449220945, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (10, N'Transformers', CAST(N'2007-07-03T00:00:00.000' AS DateTime), 18, 46, N'Michael Bay is a terrible director. But he brought back Transformers! And he got Peter Cullen to reprise his 1980s role as Optimus Prime! Sadly, he failed to cast Frank Welker as Megatron. And he hired Shia "I need a massive punch in the face" LaBoeuf and Megan "I''m Angelina Jolie without the class or talent" Fox. You suck Michael Bay.', 241, 1, 3, 144, 4, 150000000, 709709780, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (11, N'Harry Potter and the Order of the Phoenix', CAST(N'2007-07-11T00:00:00.000' AS DateTime), 19, 47, N'Not watched it but isn''t this the one where Gary Oldman dies? Killing off your best character there, bravo Rowling, bravo.', 240, 1, 12, 138, 4, 150000000, 939885929, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (12, N'Beowulf', CAST(N'2007-11-16T00:00:00.000' AS DateTime), 20, 48, N'One of the greatest literary works of Anglo-Saxon Old English is mangled by the raspy cockernee voice of a man so ugly he had to be replaced by CGI.', 241, 1, 12, 115, 4, 150000000, 196393745, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (13, N'Bee Movie', CAST(N'2007-11-02T00:00:00.000' AS DateTime), 21, 8, N'Not watched it.', 241, 1, 11, 91, 1, 150000000, 287594577, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (14, N'Pirates of the Caribbean: At World''s End', CAST(N'2007-05-25T00:00:00.000' AS DateTime), 22, 3, N'Honestly, I stopped paying attention after the first one. I think this is the one where Legolas gets stabbed but then comes back as an undead ship captain. So he''s basically turned into a zombie fisherman. Like Jesus.', 241, 1, 8, 168, 4, 300000000, 963420425, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (15, N'I Am Legend', CAST(N'2007-12-14T00:00:00.000' AS DateTime), 23, 49, N'Will Smith spends months capturing vampires to cure them instead of trying to stake them through the heart. It is thus completely his own fault that he dies in the end. Unless you watch the alternate ending in which he lives happily ever after. Because Hollywood.', 241, 1, 10, 100, 5, 150000000, 585349010, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (16, N'Ratatouille', CAST(N'2007-06-22T00:00:00.000' AS DateTime), 24, 3, N'"There''s a rat in my kitchen what am I gonna do?";
The answer, apparently, is to appoint him head chef. Who knew?', 241, 1, 11, 111, 1, 150000000, 623722818, 5, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (17, N'Troy', CAST(N'2004-05-14T00:00:00.000' AS DateTime), 25, 50, N'The single most amazing fact about this film is that Sean Bean is in it and he doesn''t die! Some of the swordfights are pretty spectacular. Other than that, it''s pretty rubbish.', 241, 1, 8, 162, 5, 175000000, 497409852, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (18, N'Harry Potter and the Goblet of Fire', CAST(N'2005-11-06T00:00:00.000' AS DateTime), 26, 47, N'Not seen it, but it''s probably rubbish.', 240, 1, 12, 156, 4, 150000000, 896911078, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (19, N'Batman Begins', CAST(N'2005-06-15T00:00:00.000' AS DateTime), 27, 43, N'Spoilt little man child Christian Bale plays spoilt little man child Bruce Wayne. Bruce begins the film on the worst gap year in history;  homeless, destitute and imprisoned. Then he gets beaten up by ninjas and you find out his mum and dad were killed when he was a little boy. You feel sorry for Bruce. Then his personal butler arrives to collect him in his private jet and takes him back to his castle and you find it impossible to sympathise with Bruce ever again.;
Fun fact: I used to live in the village of Mentmore, next door to Mentmore Towers, the huge mansion that was used as Wayne Manor in the film.', 241, 1, 4, 141, 4, 150000000, 374218673, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (20, N'Charlie and the Chocolate Factory', CAST(N'2005-07-15T00:00:00.000' AS DateTime), 28, 49, N'This film wishes it was as good as Willy Wonka and the Chocolate Factory. It isn''t. It isn''t even close.', 241, 1, 12, 115, 2, 150000000, 474968763, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (21, N'Pirates of the Caribbean: Dead Man''s Chest', CAST(N'2006-07-06T00:00:00.000' AS DateTime), 22, 3, N'It''s ok. Not as good as the first one though.', 241, 1, 8, 151, 4, 225000000, 1066179725, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (22, N'Die Another Day', CAST(N'2002-11-20T00:00:00.000' AS DateTime), 29, 51, N'Too many gadgets, not enough of Bond punching people. Away with you Brosnan, and take your silly invisible car with you.', 240, 1, 9, 133, 4, 142000000, 431971116, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (23, N'Lethal Weapon 4', CAST(N'1998-07-10T00:00:00.000' AS DateTime), 30, 52, N'The least believable film of the series, mainly due to Jet Li being beaten up by Mel Gibson. Nonsense! In Jet Li''s country, Gibson would already be dead. Also, Danny Glover is still too old for this s***.', 241, 1, 4, 127, 5, 100000000, 285444603, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (24, N'Armageddon', CAST(N'1998-07-01T00:00:00.000' AS DateTime), 18, 7, N'One of the very few films I''ve had to stop watching part way through because it was so terrible. I''ve still never seen it to the end and can only hope that the asteroid won so that everybody involved in this dreadful film died horrible, violent deaths.', 241, 1, 3, 151, 3, 140000000, 553709788, 4, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (25, N'Men in Black II', CAST(N'2002-07-03T00:00:00.000' AS DateTime), 31, 41, N'Meh. It was ok, not as good as the first.', 241, 1, 3, 89, 2, 140000000, 441818803, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (26, N'Spider-Man 3', CAST(N'2007-05-04T00:00:00.000' AS DateTime), 11, 53, N'Attempts to spice up dreary Tobey Maguire''s performance by turning him into Goth-Spiderman fail miserably. Tobey remains dreary, film remains uninteresting.', 241, 1, 4, 139, 4, 258000000, 890871626, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (27, N'Spider-Man 2', CAST(N'2004-06-30T00:00:00.000' AS DateTime), 11, 53, N'I know that I''ve watched this but I genuinely can''t remember a thing about it. I don''t know why Alfred Molina wasted his time and talent on it.', 241, 1, 4, 127, 4, 200000000, 783766341, 3, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (28, N'Stealth', CAST(N'2005-08-05T00:00:00.000' AS DateTime), 32, 54, N'Of the many things in my life for which I am grateful, the fact that I have never seen Stealth ranks right up there.', 241, 1, 4, 121, 4, 135000000, 76932872, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (29, N'Final Fantasy: The Spirits Within', CAST(N'2001-08-03T00:00:00.000' AS DateTime), 33, 55, N'Never played the game, never watched the film.', 241, 1, 11, 106, 2, 137000000, 85131830, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (30, N'Miami Vice', CAST(N'2006-07-28T00:00:00.000' AS DateTime), 34, 1, N'It didn''t get the best reviews, but I like Michael Mann''s more subdued take on the 80''s extravagance of its source material.', 241, 1, 9, 134, 5, 135000000, 163794509, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (31, N'The World Is Not Enough', CAST(N'1999-11-26T00:00:00.000' AS DateTime), 35, 51, N'Robert Carlyle can''t feel pain. Denise Richards can''t act. I can''t bear to watch this film whenever it''s on TV.', 240, 1, 9, 128, 3, 135000000, 361832400, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (32, N'Master and Commander: The Far Side of the World', CAST(N'2003-11-14T00:00:00.000' AS DateTime), 36, 4, N'I found this surprisingly entertaining. I''d watch a sequel but sadly they aren''t making one.', 241, 1, 8, 138, 4, 150000000, 212011111, 10, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (33, N'War of the Worlds', CAST(N'2005-06-29T00:00:00.000' AS DateTime), 4, 41, N'I watched this as a relatively dumb, big-budget sci-fi flick. I''ve since realised that it''s an intelligent and damning critique of the immense stupidity of the anti-vaccination crowd. On an unrelated note, did we ever find out what happened to Tom Cruise''s sweet ''68 Ford Mustang?', 241, 1, 3, 116, 4, 132000000, 591745540, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (34, N'The Bourne Ultimatum', CAST(N'2007-08-17T00:00:00.000' AS DateTime), 37, 56, N'It''s still good, but it lacks the impact that the first film had. Except for the part where Matt Damon jumps down a stairwell onto a fat guy. That was a pretty good impact.', 241, 1, 9, 115, 4, 110000000, 442824138, 3, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (35, N'Harry Potter and the Prisoner of Azkaban', CAST(N'2004-05-31T00:00:00.000' AS DateTime), 38, 47, N'They definitely made too many of these films. Not seen it, don''t care.', 240, 1, 12, 141, 2, 130000000, 796688549, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (36, N'The Matrix Reloaded', CAST(N'2003-05-15T00:00:00.000' AS DateTime), 39, 49, N'I kind of feel sorry for the Wachowskis. They''d made the absolute perfect sci-fi movie in The Matrix, how could a sequel ever hope to live up to the original''s reputation? It didn''t. Still worth a watch though. Unlike the third one.', 241, 1, 3, 138, 5, 150000000, 742128461, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (37, N'The Da Vinci Code', CAST(N'2006-05-19T00:00:00.000' AS DateTime), 40, 20, N'I have watched this and it still amazes me how so many talented people can produce a film that''s just so "meh".', 241, 1, 18, 146, 4, 125000000, 758239851, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (38, N'Harry Potter and the Philosopher''s Stone', CAST(N'2001-11-16T00:00:00.000' AS DateTime), 41, 47, N'Filmum es stinkum! Away with you Potter and take your faux-Latin voodoo with you. If I wanted to see a bunch of amateurish child actors stumbling over lines of basic English I''d go and watch my nephew''s school play.', 240, 1, 12, 152, 2, 125000000, 974800000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (39, N'The Day After Tomorrow', CAST(N'2004-05-26T00:00:00.000' AS DateTime), 42, 57, N'Dennis Quaid saves the world. The End.', 241, 1, 9, 124, 4, 125000000, 542772402, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (40, N'Pirates of the Caribbean: Curse of the Black Pearl', CAST(N'2003-07-09T00:00:00.000' AS DateTime), 22, 3, N'Forget the ever-growing nonsense of the other films in the series - this first one is great. The story is as sophisticated as you''d expect from a film based on a fairground ride but it''s just as entertaining.', 241, 1, 8, 135, 3, 140000000, 654264015, 5, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (41, N'The Last Samurai', CAST(N'2003-12-05T00:00:00.000' AS DateTime), 43, 58, N'I desperately want to hate this film for its reliance on borderline-racist tropes: the mystical savage, the noble white hunter, it''s basically Dances with Wolves in Japan. However, I adore samurai movies and this is a spectacular one, as much for its quiet, introspective moments as for the blood and fury battle scenes. Reading around the subject of the Meiji Restoration and the Satsuma Rebellion is highly recommended to get the most out of this film.', 241, 1, 8, 154, 5, 140000000, 456758981, 4, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (42, N'The Perfect Storm', CAST(N'2000-06-30T00:00:00.000' AS DateTime), 25, 6, N'Tiny Boat vs. Enormous Angry Ocean. Boat loses.', 241, 1, 8, 130, 3, 120000000, 328718434, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (43, N'Mission: Impossible III', CAST(N'2006-05-04T00:00:00.000' AS DateTime), 44, 59, N'I have seen it but all I can remember is the very beginning and the very end. Nothing inbetween. It''s probably ok.', 241, 1, 9, 125, 4, 150000000, 397850012, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (44, N'Mission: Impossible II', CAST(N'2000-05-24T00:00:00.000' AS DateTime), 45, 59, N'You can tell that John Woo directed this because at one point there is a joust. Using motorbikes.', 241, 1, 9, 123, 5, 125000000, 546388105, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (45, N'Windtalkers', CAST(N'2002-08-30T00:00:00.000' AS DateTime), 45, 60, N'I''ve seen it but can''t remember it. John Woo directed though, so there''s probably a dove in there somewhere.', 241, 1, 5, 134, 5, 115000000, 77628265, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (46, N'Star Wars: Episode II - Attack of the Clones', CAST(N'2002-05-16T00:00:00.000' AS DateTime), 7, 13, N'Oh god, just when you thought that nobody could be worse than the horrible little child who played Anakin Skywalker in the previous film they get Hayden Christensen - a man so untalented as an actor that he delivered his lines as though he hadn''t quite learned to read. Dreadfull, dreadful, dreadful.', 241, 1, 3, 142, 2, 115000000, 649400000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (47, N'Monsters, Inc.', CAST(N'2002-02-08T00:00:00.000' AS DateTime), 46, 10, N'Ahh, just adorable. Some people think that you need to have children in order to justify watching films like this. You don''t.', 241, 1, 11, 92, 1, 115000000, 562800000, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (48, N'Star Wars: Episode I - The Phantom Menace', CAST(N'1999-05-19T00:00:00.000' AS DateTime), 7, 13, N'I had such high hopes for this. I remember watching the online trailer over and over in the tiny window of Apple Quicktime Player on my mate''s PC over the dodgy university network. How were we to know that the trailer would be infinity times better than the actual film. A sad, sad day. The sole redeeming feature is the best lightsaber duel of all of the films so far, between an actual real martial artist in Ray Park and Ewan McGregor. The rest of it? I still try to pretend that it doesn''t exist.', 241, 1, 3, 133, 1, 115000000, 1027000000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (49, N'Star Wars: Episode III - Revenge of the Sith', CAST(N'2005-05-19T00:00:00.000' AS DateTime), 7, 13, N'When Hayden Christensen attempts to act, he looks as though he''s trying to understand advanced string theory. I think he''d make a more believable theoretical physicist than he does an actor. The only good thing about this film is that it''s the final Star Wars movie in which George Lucas had any sort of control.', 241, 1, 3, 140, 4, 113000000, 848800000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (50, N'The Matrix Revolutions', CAST(N'2003-11-05T00:00:00.000' AS DateTime), 39, 49, N'Just... No.', 241, 1, 3, 129, 5, 150000000, 427343298, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (51, N'Tomorrow Never Dies', CAST(N'1997-12-12T00:00:00.000' AS DateTime), 47, 51, N'Apart from Live and Let Die and Moonraker, this must be my most watched Bond film. Not because I particularly like it, but because it''s the only one they ever seem to show on British TV.', 240, 1, 9, 119, 3, 110000000, 333011068, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (52, N'Ocean''s Eleven', CAST(N'2002-02-15T00:00:00.000' AS DateTime), 48, 61, N'I don''t remember anything about this film.', 241, 1, 14, 117, 3, 85000000, 450717150, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (53, N'Live Free or Die Hard', CAST(N'2007-06-27T00:00:00.000' AS DateTime), 49, 62, N'The best thing about this film is that Justin Long plays the geeky, annoying sidekick. It could have been worse, it could have been Shia LaBoeuf.', 241, 1, 4, 130, 5, 110000000, 383500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (54, N'Around the World in 80 Days', CAST(N'2004-06-16T00:00:00.000' AS DateTime), 50, 63, N'I adore Jackie Chan movies and I think Steve Coogan is great, but even I won''t watch this.', 241, 1, 8, 120, 2, 110000000, 72178895, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (55, N'Kingdom of Heaven', CAST(N'2005-05-06T00:00:00.000' AS DateTime), 51, 64, NULL, 241, 1, 8, 144, 5, 130000000, 211398413, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (56, N'Mr. & Mrs. Smith', CAST(N'2005-06-10T00:00:00.000' AS DateTime), 52, 65, NULL, 241, 1, 4, 120, 5, 110000000, 478336279, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (57, N'The Aviator', CAST(N'2004-12-25T00:00:00.000' AS DateTime), 9, 66, N'Haven''t seen it.', 241, 1, 19, 170, 4, 110000000, 214608827, 11, 5);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (58, N'Ali', CAST(N'2001-12-25T00:00:00.000' AS DateTime), 34, 67, NULL, 241, 1, 19, 157, 5, 107000000, 87713825, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (59, N'The Lord of the Rings: Fellowship of the Ring', CAST(N'2001-12-19T00:00:00.000' AS DateTime), 12, 42, N'If you can get over the painfully earnest performances and you''re not one of the Tolkien fanboys who complained about all the changes from the book this is a great, epic adaptation and thoroughly entertaining.', 167, 1, 12, 178, 2, 93000000, 871530324, 13, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (60, N'I, Robot', CAST(N'2004-07-15T00:00:00.000' AS DateTime), 53, 68, N'Will Smith''s lovely day with his new Converse All Stars is ruined by a bunch of killer robots controlled by a maniacal AI. It all works out in the end though, apart from the fact that Shia LaBoeuf is still alive at the end of the film.', 241, 1, 3, 115, 4, 120000000, 347234916, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (61, N'Casino Royale', CAST(N'2006-11-14T00:00:00.000' AS DateTime), 54, 51, N'Remember when they announced that Daniel Craig was going to be the next Bond and everyone went mental because they knew he was going to be terrible and blond and just a rubbish Bond? Not me, I said he was going to be awesome and I was right.', 240, 1, 9, 144, 4, 150000000, 599000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (62, N'Minority Report', CAST(N'2002-07-04T00:00:00.000' AS DateTime), 4, 41, NULL, 241, 1, 3, 145, 3, 102000000, 358300000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (63, N'Terminator 2: Judgement Day', CAST(N'1991-08-16T00:00:00.000' AS DateTime), 15, 15, NULL, 241, 1, 3, 136, 5, 100000000, 516816151, 4, 6);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (64, N'Catwoman', CAST(N'2004-08-13T00:00:00.000' AS DateTime), 55, 49, NULL, 241, 1, 4, 104, 4, 100000000, 82102379, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (65, N'Harry Potter and the Chamber of Secrets', CAST(N'2002-11-15T00:00:00.000' AS DateTime), 41, 47, N'Richard Harris hated this film so much that he died so as not to be cast in the next one.', 240, 1, 12, 155, 2, 100000000, 878979634, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (66, N'American Gangster', CAST(N'2007-11-16T00:00:00.000' AS DateTime), 51, 20, NULL, 241, 1, 19, 158, 6, 100000000, 266465037, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (67, N'Blood Diamond', CAST(N'2006-12-08T00:00:00.000' AS DateTime), 43, 69, NULL, 241, 1, 8, 143, 5, 100000000, 171407179, 5, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (68, N'Gladiator', CAST(N'2000-05-05T00:00:00.000' AS DateTime), 51, 64, NULL, 241, 1, 8, 155, 5, 103000000, 457600000, 12, 5);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (69, N'True Lies', CAST(N'1994-07-15T00:00:00.000' AS DateTime), 15, 70, NULL, 241, 1, 9, 141, 5, 100000000, 378882411, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (70, N'Gangs of New York', CAST(N'2002-12-20T00:00:00.000' AS DateTime), 9, 7, N'I still can''t believe that this film didn''t win any Oscars. But at least Scorsese still has his integrity and talent - reportedly George Lucas visited the set and told Scorsese that he could have created it all with CGI. And that''s why George Lucas sucks.', 241, 1, 2, 160, 6, 97000000, 193772504, 10, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (71, N'Black Hawk Down', CAST(N'2002-01-18T00:00:00.000' AS DateTime), 51, 19, NULL, 241, 1, 5, 144, 5, 92000000, 172989651, 4, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (72, N'The Fifth Element', CAST(N'1997-05-07T00:00:00.000' AS DateTime), 56, 18, NULL, 79, 1, 3, 127, 2, 90000000, 263920180, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (73, N'The Lord of the Rings: Return of the King', CAST(N'2003-12-17T00:00:00.000' AS DateTime), 12, 42, N'Now this is how you drag out an ending!', 167, 1, 12, 201, 4, 94000000, 1119929521, 11, 11);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (74, N'The Lord of the Rings: The Two Towers', CAST(N'2002-12-18T00:00:00.000' AS DateTime), 12, 42, NULL, 167, 1, 12, 179, 4, 94000000, 926047111, 6, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (75, N'Rush Hour 3', CAST(N'2007-08-10T00:00:00.000' AS DateTime), 57, 71, NULL, 241, 1, 17, 91, 4, 140000000, 258022233, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (76, N'Rush Hour 2', CAST(N'2001-08-03T00:00:00.000' AS DateTime), 57, 14, NULL, 241, 1, 17, 90, 3, 90000000, 347425832, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (77, N'Men in Black', CAST(N'1997-07-02T00:00:00.000' AS DateTime), 31, 41, NULL, 241, 1, 3, 98, 2, 90000000, 589390539, 3, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (78, N'The Departed', CAST(N'2006-10-06T00:00:00.000' AS DateTime), 9, 6, N'The film that finally won Scorsese his Oscar. Long overdue in my opinion. Mark Wahlberg should have won one for most epic swearing in a feature film.', 241, 1, 9, 151, 6, 90000000, 289847354, 5, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (79, N'Die Hard: With A Vengeance', CAST(N'1995-08-18T00:00:00.000' AS DateTime), 58, 72, NULL, 241, 1, 4, 131, 5, 90000000, 366101666, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (80, N'Cinderella Man', CAST(N'2005-09-09T00:00:00.000' AS DateTime), 40, 20, N'Haven''t seen it.', 241, 1, 2, 144, 4, 88000000, 108539911, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (81, N'The Bourne Supremacy', CAST(N'2004-07-23T00:00:00.000' AS DateTime), 37, 56, NULL, 241, 1, 9, 108, 4, 75000000, 288500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (82, N'Last Action Hero', CAST(N'1993-07-30T00:00:00.000' AS DateTime), 58, 5, NULL, 241, 1, 4, 130, 5, 85000000, 137298489, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (83, N'Cast Away', CAST(N'2001-01-12T00:00:00.000' AS DateTime), 20, 9, NULL, 241, 1, 2, 143, 3, 85000000, 427230516, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (84, N'Ocean''s Twelve', CAST(N'2005-02-04T00:00:00.000' AS DateTime), 48, 61, NULL, 241, 1, 14, 125, 4, 110000000, 362744280, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (85, N'Click', CAST(N'2006-09-29T00:00:00.000' AS DateTime), 50, 19, N'Adam Sandler fast forwards his life. In much the same way as I fast forward most of his films rather than watch them.', 241, 1, 7, 107, 4, 82500000, 237681299, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (86, N'Bruce Almighty', CAST(N'2003-05-23T00:00:00.000' AS DateTime), 16, 44, NULL, 241, 1, 7, 101, 4, 81000000, 484592874, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (87, N'Mission: Impossible', CAST(N'1996-05-22T00:00:00.000' AS DateTime), 59, 59, NULL, 241, 1, 9, 110, 2, 80000000, 457696359, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (88, N'Deep Impact', CAST(N'1998-05-08T00:00:00.000' AS DateTime), 60, 8, N'Of the two asteroid-themed disaster movies released this year this was by far the least bad. Note that comparing a film favourably to Armageddon isn''t actually a compliment.', 241, 1, 3, 121, 3, 80000000, 349464664, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (89, N'Road to Perdition', CAST(N'2002-07-12T00:00:00.000' AS DateTime), 61, 73, NULL, 241, 1, 14, 117, 5, 80000000, 181000000, 6, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (90, N'A Beautiful Mind', CAST(N'2001-12-21T00:00:00.000' AS DateTime), 40, 20, N'Haven''t seen it.', 241, 1, 2, 135, 3, 58000000, 313542341, 8, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (91, N'Children of Men', CAST(N'2006-09-22T00:00:00.000' AS DateTime), 38, 74, NULL, 241, 1, 3, 109, 5, 76000000, 69900000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (92, N'Munich', CAST(N'2005-12-23T00:00:00.000' AS DateTime), 4, 41, NULL, 241, 1, 2, 163, 5, 77000000, 130358911, 5, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (93, N'Independence Day', CAST(N'1996-07-02T00:00:00.000' AS DateTime), 42, 57, NULL, 241, 1, 3, 145, 3, 75000000, 817400891, 2, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (94, N'X-Men', CAST(N'2000-08-18T00:00:00.000' AS DateTime), 14, 4, N'The film that single-handedly sparked the comic book movie revolution. I still think that this is the best of the X-Men films just because it arrived so unexpectedly and without fanfare.', 241, 1, 4, 104, 3, 75000000, 296339527, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (95, N'X2', CAST(N'2003-05-02T00:00:00.000' AS DateTime), 14, 4, NULL, 241, 1, 4, 134, 4, 110000000, 407711549, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (96, N'The Rock', CAST(N'1996-06-07T00:00:00.000' AS DateTime), 18, 75, NULL, 241, 1, 9, 136, 5, 75000000, 335100000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (97, N'Constantine', CAST(N'2005-02-18T00:00:00.000' AS DateTime), 23, 49, NULL, 241, 1, 9, 121, 5, 100000000, 230884728, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (98, N'Unbreakable', CAST(N'2000-12-29T00:00:00.000' AS DateTime), 62, 7, NULL, 241, 1, 2, 106, 3, 75000000, 248118121, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (99, N'Talladega Nights: The Ballad of Ricky Bobby', CAST(N'2006-09-15T00:00:00.000' AS DateTime), 63, 76, NULL, 241, 1, 7, 108, 4, 72500000, 162966177, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (100, N'The Simpsons Movie', CAST(N'2007-07-25T00:00:00.000' AS DateTime), 64, 77, N'I think that only one episodic cartoon show has successfully made the leap from small to big screen. That show is South Park.', 241, 1, 11, 87, 2, 72000000, 527000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (101, N'Braveheart', CAST(N'1995-05-24T00:00:00.000' AS DateTime), 65, 25, NULL, 241, 1, 13, 177, 5, 72000000, 210409945, 10, 5);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (102, N'Jarhead', CAST(N'2006-01-13T00:00:00.000' AS DateTime), 61, 1, NULL, 241, 1, 5, 125, 5, 72000000, 96889998, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (103, N'The Village', CAST(N'2004-08-20T00:00:00.000' AS DateTime), 62, 7, NULL, 241, 1, 9, 108, 4, 60000000, 256697520, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (104, N'Shrek the Third', CAST(N'2007-05-18T00:00:00.000' AS DateTime), 66, 8, NULL, 241, 1, 11, 93, 1, 160000000, 799000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (105, N'Shrek 2', CAST(N'2004-05-19T00:00:00.000' AS DateTime), 67, 8, NULL, 241, 1, 11, 93, 1, 150000000, 919838758, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (106, N'The Abyss', CAST(N'1989-08-09T00:00:00.000' AS DateTime), 15, 70, NULL, 241, 1, 3, 140, 5, 70000000, 90000098, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (107, N'Star Trek: Insurrection', CAST(N'1998-12-11T00:00:00.000' AS DateTime), 68, 2, NULL, 241, 1, 3, 103, 2, 58000000, 112587658, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (108, N'Who Framed Roger Rabbit?', CAST(N'1988-06-22T00:00:00.000' AS DateTime), 20, 7, N'It was Doc Brown.', 241, 1, 7, 103, 2, 70000000, 329800000, 6, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (109, N'Sleepy Hollow', CAST(N'1999-11-19T00:00:00.000' AS DateTime), 28, 78, NULL, 241, 1, 10, 105, 5, 100000000, 206071502, 3, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (110, N'xXx', CAST(N'2002-08-09T00:00:00.000' AS DateTime), 32, 19, NULL, 241, 1, 4, 124, 5, 70000000, 277448382, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (111, N'Die Hard 2', CAST(N'1990-07-04T00:00:00.000' AS DateTime), 69, 52, NULL, 241, 1, 4, 124, 5, 70000000, 240031094, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (112, N'The Long Kiss Goodnight', CAST(N'1996-11-29T00:00:00.000' AS DateTime), 69, 14, N'A surprising lack of kissing in this movie. More than made up for by all the swearing, violence and explosions.', 241, 1, 4, 120, 6, 65000000, 89456761, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (113, N'Apollo 13', CAST(N'1995-06-30T00:00:00.000' AS DateTime), 40, 20, NULL, 241, 1, 13, 140, 2, 52000000, 355237933, 9, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (114, N'Saving Private Ryan', CAST(N'1998-07-24T00:00:00.000' AS DateTime), 4, 41, NULL, 241, 1, 5, 170, 5, 70000000, 481840909, 11, 5);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (115, N'Fight Club', CAST(N'1999-10-15T00:00:00.000' AS DateTime), 70, 79, N'Sorry, I can''t talk about it.', 241, 1, 9, 139, 6, 63000000, 100900000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (116, N'The Matrix', CAST(N'1999-06-11T00:00:00.000' AS DateTime), 39, 49, N'Remember when this was the best, most mind-blowing, futuristic thing you''d ever seen? My mate had one of those Nokia flip phones which instantly transformed him into the coolest person at uni.', 241, 1, 3, 136, 5, 65000000, 463500000, 4, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (117, N'Total Recall', CAST(N'1990-06-01T00:00:00.000' AS DateTime), 72, 15, N'If you haven''t watched it for a while it''s easy to forget how brutally violent this film is. It''s still a great piece of pulp sci-fi and significantly better than the remake.', 241, 1, 3, 113, 6, 60000000, 261300000, 2, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (118, N'Cliffhanger', CAST(N'1993-06-25T00:00:00.000' AS DateTime), 69, 15, NULL, 241, 1, 4, 113, 5, 65000000, 255325036, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (119, N'Clear and Present Danger', CAST(N'1994-08-03T00:00:00.000' AS DateTime), 73, 2, NULL, 241, 1, 9, 141, 3, 62000000, 215887717, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (120, N'GoldenEye', CAST(N'1995-11-24T00:00:00.000' AS DateTime), 54, 51, N'I like how Sean Bean can''t hide his Sheffield accent when he gets angry.', 240, 1, 9, 130, 3, 58000000, 352194034, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (121, N'The Black Dahlia', CAST(N'2006-09-15T00:00:00.000' AS DateTime), 59, 80, NULL, 241, 1, 14, 122, 5, 50000000, 49332692, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (122, N'The Truman Show', CAST(N'1998-06-05T00:00:00.000' AS DateTime), 36, 81, N'Well waddya know, Jim Carrey can actually act!', 241, 1, 2, 103, 2, 60000000, 264118201, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (123, N'Event Horizon', CAST(N'1997-08-15T00:00:00.000' AS DateTime), 74, 2, N'This film scares the bejesus out of me every single time but I still love it. It''s pretty much the only horror movie that I''ll watch.', 241, 1, 3, 96, 6, 60000000, 47073851, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (124, N'The Bourne Identity', CAST(N'2002-06-14T00:00:00.000' AS DateTime), 52, 56, NULL, 241, 1, 9, 118, 4, 60000000, 214034224, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (125, N'Hellboy', CAST(N'2004-04-02T00:00:00.000' AS DateTime), 71, 19, NULL, 241, 1, 4, 122, 4, 66000000, 99318987, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (126, N'Starsky & Hutch', CAST(N'2004-03-19T00:00:00.000' AS DateTime), 75, 82, NULL, 241, 1, 7, 101, 5, 60000000, 170268750, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (127, N'Intolerable Cruelty', CAST(N'2003-10-10T00:00:00.000' AS DateTime), 5, 20, NULL, 241, 1, 7, 100, 4, 60000000, 120217409, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (128, N'300', CAST(N'2007-03-23T00:00:00.000' AS DateTime), 76, 43, NULL, 241, 1, 8, 117, 5, 65000000, 456000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (129, N'Star Trek: Nemesis', CAST(N'2003-01-03T00:00:00.000' AS DateTime), 77, 2, NULL, 241, 1, 3, 116, 4, 60000000, 67312826, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (130, N'Superman', CAST(N'1978-12-10T00:00:00.000' AS DateTime), 30, 83, NULL, 241, 1, 8, 143, 2, 55000000, 300218018, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (131, N'Crimson Tide', CAST(N'1995-05-12T00:00:00.000' AS DateTime), 78, 75, NULL, 241, 1, 9, 116, 5, 53000000, 157387195, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (132, N'Deja Vu', CAST(N'2006-11-22T00:00:00.000' AS DateTime), 78, 7, N'I''m sure I''ve already seen this…', 241, 1, 9, 126, 4, 75000000, 180557550, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (133, N'Man On Fire', CAST(N'2004-04-21T00:00:00.000' AS DateTime), 78, 65, NULL, 241, 1, 9, 146, 6, 70000000, 130293714, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (134, N'Enemy of the State', CAST(N'1998-12-26T00:00:00.000' AS DateTime), 78, 7, NULL, 241, 1, 9, 132, 5, 90000000, 250649836, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (135, N'Kill Bill: Vol. 1', CAST(N'2003-10-17T00:00:00.000' AS DateTime), 79, 84, NULL, 241, 1, 17, 111, 6, 30000000, 180900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (136, N'Alien 3', CAST(N'1992-05-22T00:00:00.000' AS DateTime), 70, 85, NULL, 241, 1, 3, 115, 6, 55000000, 159800000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (137, N'Shanghai Noon', CAST(N'2000-05-26T00:00:00.000' AS DateTime), 80, 7, NULL, 241, 1, 17, 110, 3, 55000000, 99274467, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (138, N'Kill Bill: Vol. 2', CAST(N'2004-04-23T00:00:00.000' AS DateTime), 79, 84, NULL, 241, 1, 17, 136, 6, 30000000, 152159461, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (139, N'Blade II', CAST(N'2002-03-22T00:00:00.000' AS DateTime), 71, 14, NULL, 241, 1, 4, 116, 6, 50000000, 155010032, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (140, N'Flags of our Fathers', CAST(N'2006-10-20T00:00:00.000' AS DateTime), 10, 2, NULL, 241, 1, 5, 132, 5, 55000000, 65900249, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (141, N'Casino', CAST(N'1996-02-23T00:00:00.000' AS DateTime), 9, 1, NULL, 241, 1, 14, 178, 6, 50000000, 116112375, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (142, N'Robin Hood: Prince of Thieves', CAST(N'1991-06-14T00:00:00.000' AS DateTime), 17, 21, NULL, 241, 1, 8, 155, 2, 48000000, 390493908, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (143, N'Shrek', CAST(N'2001-05-18T00:00:00.000' AS DateTime), 67, 86, NULL, 241, 1, 11, 90, 1, 60000000, 484409218, 2, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (144, N'Shanghai Knights', CAST(N'2003-02-07T00:00:00.000' AS DateTime), 81, 7, NULL, 241, 1, 17, 114, 4, 50000000, 88323487, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (145, N'Indiana Jones and the Last Crusade', CAST(N'1989-05-24T00:00:00.000' AS DateTime), 4, 13, NULL, 241, 1, 8, 127, 2, 48000000, 474200000, 3, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (146, N'Star Trek: First Contact', CAST(N'1996-11-22T00:00:00.000' AS DateTime), 68, 2, NULL, 241, 1, 3, 111, 3, 45000000, 146027888, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (147, N'Blade', CAST(N'1998-08-21T00:00:00.000' AS DateTime), 82, 14, NULL, 241, 1, 12, 120, 6, 40000000, 131183530, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (148, N'Patriot Games', CAST(N'1992-06-05T00:00:00.000' AS DateTime), 73, 2, NULL, 241, 1, 9, 117, 5, 45000000, 178051587, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (149, N'The Fugitive', CAST(N'1993-08-06T00:00:00.000' AS DateTime), 83, 6, NULL, 241, 1, 9, 130, 3, 44000000, 368875760, 7, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (150, N'The Prestige', CAST(N'2006-10-20T00:00:00.000' AS DateTime), 27, 7, NULL, 241, 1, 18, 130, 4, 40000000, 109700000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (151, N'The Last of the Mohicans', CAST(N'1992-09-25T00:00:00.000' AS DateTime), 34, 21, NULL, 241, 1, 8, 112, 3, 40000000, 75505856, 1, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (152, N'Apocalypto', CAST(N'2006-12-08T00:00:00.000' AS DateTime), 65, 7, NULL, 241, 8, 8, 138, 6, 40000000, 120600000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (153, N'Wedding Crashers', CAST(N'2005-07-15T00:00:00.000' AS DateTime), 81, 87, NULL, 241, 1, 7, 119, 5, 40000000, 285176741, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (154, N'Back to the Future Part III', CAST(N'1990-05-25T00:00:00.000' AS DateTime), 20, 41, NULL, 241, 1, 3, 118, 2, 40000000, 244500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (155, N'Back to the Future Part II', CAST(N'1989-11-22T00:00:00.000' AS DateTime), 20, 41, NULL, 241, 1, 3, 108, 2, 40000000, 331900000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (156, N'Licence to Kill', CAST(N'1989-06-13T00:00:00.000' AS DateTime), 84, 51, NULL, 240, 1, 9, 133, 5, 32000000, 156000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (157, N'The Living Daylights', CAST(N'1987-06-27T00:00:00.000' AS DateTime), 84, 51, NULL, 240, 1, 9, 131, 2, 40000000, 191200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (158, N'Sin City', CAST(N'2005-04-01T00:00:00.000' AS DateTime), 85, 23, NULL, 241, 1, 14, 124, 6, 40000000, 158800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (159, N'Serenity', CAST(N'2005-10-07T00:00:00.000' AS DateTime), 86, 1, N'The single worst decision in television history saw the cancellation of Firefly after just one fantastic season. Thanks a lot Fox. Jerks. By way of consolation we got this pretty great movie offering some closure. It''s still not enough. "I''m a leaf on the wind" :''(', 241, 1, 3, 119, 5, 39000000, 38869464, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (160, N'Star Trek: Generations', CAST(N'1994-11-18T00:00:00.000' AS DateTime), 87, 2, NULL, 241, 1, 3, 118, 2, 35000000, 118071125, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (161, N'Never Say Never Again', CAST(N'1983-10-07T00:00:00.000' AS DateTime), 88, 88, NULL, 241, 1, 9, 134, 2, 36000000, 160000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (162, N'Star Trek: The Motion Picture', CAST(N'1979-12-07T00:00:00.000' AS DateTime), 89, 2, NULL, 241, 1, 3, 132, 1, 46000000, 139000000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (163, N'Rush Hour', CAST(N'1998-12-04T00:00:00.000' AS DateTime), 57, 71, NULL, 241, 1, 4, 97, 5, 33000000, 244386864, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (164, N'Lethal Weapon 3', CAST(N'1992-08-14T00:00:00.000' AS DateTime), 30, 52, NULL, 241, 1, 4, 118, 5, 35000000, 321731527, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (165, N'Star Wars: Episode VI - Return of the Jedi', CAST(N'1983-05-25T00:00:00.000' AS DateTime), 90, 13, NULL, 241, 1, 3, 132, 1, 42700000, 475100000, 4, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (166, N'A History of Violence', CAST(N'2005-09-23T00:00:00.000' AS DateTime), 91, 14, NULL, 241, 1, 2, 96, 6, 32000000, 60740827, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (167, N'Moonraker', CAST(N'1979-06-29T00:00:00.000' AS DateTime), 92, 51, NULL, 240, 1, 9, 126, 2, 34000000, 210300000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (168, N'Get Shorty', CAST(N'1995-10-20T00:00:00.000' AS DateTime), 31, 89, NULL, 241, 1, 14, 105, 5, 30250000, 115101622, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (169, N'The Assassination of Jesse James by the Coward Robert Ford', CAST(N'2007-09-21T00:00:00.000' AS DateTime), 93, 90, NULL, 241, 1, 1, 159, 5, 30000000, 15000000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (170, N'Million Dollar Baby', CAST(N'2005-01-14T00:00:00.000' AS DateTime), 10, 91, N'I was enjoying this, unaware of the horrendously depressing "twist" towards the end of the tale. I''ve come to realise that I really hate Clint Eastwood as a director.', 241, 1, 22, 132, 4, 30000000, 216800000, 7, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (171, N'Seven', CAST(N'1995-09-22T00:00:00.000' AS DateTime), 70, 14, NULL, 241, 1, 9, 127, 6, 33000000, 327300000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (172, N'The Hunt For Red October', CAST(N'1990-03-02T00:00:00.000' AS DateTime), 58, 92, N'Why is a Scotchman captaining a Russian nuclear submarine? Actually, I don''t care, this is a pretty great movie anyway.', 241, 1, 9, 134, 2, 30000000, 200512643, 3, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (173, N'Ghostbusters', CAST(N'1984-06-07T00:00:00.000' AS DateTime), 94, 93, NULL, 241, 1, 7, 107, 2, 30000000, 295200000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (174, N'Star Trek V: The Final Frontier', CAST(N'1989-06-09T00:00:00.000' AS DateTime), 95, 2, N'The one where a spaceship shoots god in the face.', 241, 1, 3, 106, 2, 33000000, 63000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (175, N'Indiana Jones and the Temple of Doom', CAST(N'1984-05-23T00:00:00.000' AS DateTime), 4, 13, NULL, 241, 1, 8, 118, 3, 28170000, 333107271, 2, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (176, N'Blade Runner', CAST(N'1982-06-25T00:00:00.000' AS DateTime), 51, 94, NULL, 241, 1, 3, 116, 5, 28000000, 33800000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (177, N'Die Hard', CAST(N'1988-07-15T00:00:00.000' AS DateTime), 58, 52, NULL, 241, 1, 4, 132, 5, 28000000, 140700000, 4, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (178, N'For Your Eyes Only', CAST(N'1981-06-24T00:00:00.000' AS DateTime), 84, 51, NULL, 240, 1, 9, 127, 2, 28000000, 194900000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (179, N'Octopussy', CAST(N'1983-06-06T00:00:00.000' AS DateTime), 84, 51, NULL, 241, 1, 9, 131, 2, 27500000, 183700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (180, N'Star Trek VI: The Undiscovered Country', CAST(N'1991-12-06T00:00:00.000' AS DateTime), 96, 2, NULL, 241, 1, 3, 113, 2, 27000000, 96888996, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (181, N'The Running Man', CAST(N'1987-11-13T00:00:00.000' AS DateTime), 97, 95, N'It''s like Hunger Games but infinitely better in every conceivable way.', 241, 1, 3, 101, 6, 27000000, 38122105, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (182, N'Open Range', CAST(N'2003-08-15T00:00:00.000' AS DateTime), 98, 7, NULL, 241, 1, 1, 139, 4, 22000000, 68296293, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (183, N'Kindergarten Cop', CAST(N'1991-02-01T00:00:00.000' AS DateTime), 94, 20, NULL, 241, 1, 7, 111, 5, 15000000, 201957688, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (184, N'O Brother, Where Art Thou?', CAST(N'2000-09-15T00:00:00.000' AS DateTime), 5, 7, NULL, 241, 1, 7, 107, 3, 26000000, 71900000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (185, N'Goodfellas', CAST(N'1990-09-19T00:00:00.000' AS DateTime), 9, 6, NULL, 241, 1, 14, 146, 6, 25000000, 46800000, 6, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (186, N'The Passion of the Christ', CAST(N'2004-03-26T00:00:00.000' AS DateTime), 65, 25, N'It''s kinda like Saw for Christians.', 241, 7, 2, 126, 6, 30000000, 611900000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (187, N'The Untouchables', CAST(N'1987-06-03T00:00:00.000' AS DateTime), 59, 2, NULL, 241, 1, 14, 119, 5, 25000000, 106240936, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (188, N'Schindler''s List', CAST(N'1993-12-15T00:00:00.000' AS DateTime), 4, 41, NULL, 241, 1, 13, 197, 5, 22000000, 321200000, 12, 7);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (189, N'Anchorman: The Legend of Ron Burgundy', CAST(N'2004-09-10T00:00:00.000' AS DateTime), 63, 76, NULL, 241, 1, 7, 95, 4, 26000000, 90600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (190, N'Scarface', CAST(N'1983-12-09T00:00:00.000' AS DateTime), 59, 1, NULL, 241, 1, 14, 170, 6, 25000000, 65900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (191, N'Star Trek IV: The Voyage Home', CAST(N'1986-11-26T00:00:00.000' AS DateTime), 99, 2, NULL, 241, 1, 3, 122, 2, 21000000, 133000000, 4, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (192, N'King Kong', CAST(N'1976-12-17T00:00:00.000' AS DateTime), 100, 96, NULL, 241, 1, 8, 134, 2, 24000000, 90614445, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (193, N'Star Wars: Episode V - The Empire Strikes Back', CAST(N'1980-05-21T00:00:00.000' AS DateTime), 88, 13, NULL, 241, 1, 3, 124, 1, 33000000, 538400000, 4, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (194, N'Gandhi', CAST(N'1982-12-02T00:00:00.000' AS DateTime), 101, 97, N'A film which bemoans the imperial occupation of India while simultaneously casting a white Englishman as that nation''s heroic icon.', 240, 1, 19, 188, 2, 22000000, 52767889, 11, 8);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (195, N'Equilibrium', CAST(N'2002-12-06T00:00:00.000' AS DateTime), 102, 23, NULL, 241, 1, 3, 107, 5, 20000000, 5345000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (196, N'From Dusk Till Dawn', CAST(N'1996-01-19T00:00:00.000' AS DateTime), 103, 23, NULL, 241, 1, 10, 108, 6, 20000000, 25728961, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (197, N'Raiders of the Lost Ark', CAST(N'1981-06-12T00:00:00.000' AS DateTime), 4, 13, NULL, 241, 1, 8, 115, 2, 18000000, 389900000, 9, 5);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (199, N'Back to the Future', CAST(N'1985-07-03T00:00:00.000' AS DateTime), 20, 41, NULL, 241, 1, 3, 116, 2, 19000000, 392000000, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (200, N'Dances with Wolves', CAST(N'1990-10-19T00:00:00.000' AS DateTime), 98, 27, NULL, 241, 1, 1, 180, 5, 2000000, 424208848, 12, 7);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (201, N'Star Trek III: The Search For Spock', CAST(N'1984-06-01T00:00:00.000' AS DateTime), 99, 2, NULL, 241, 1, 3, 105, 2, 16000000, 87000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (202, N'Borat: Cultural Learnings of America for Make Benefit Glorious Nation of Kazakhstan', CAST(N'2006-11-02T00:00:00.000' AS DateTime), 104, 28, NULL, 240, 1, 7, 84, 5, 18000000, 261600000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (203, N'Raging Bull', CAST(N'1980-11-14T00:00:00.000' AS DateTime), 9, 22, NULL, 241, 1, 22, 129, 6, 18000000, 23400000, 8, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (204, N'Aliens', CAST(N'1986-07-18T00:00:00.000' AS DateTime), 15, 85, NULL, 241, 1, 3, 137, 6, 18000000, 183300000, 7, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (205, N'Hero', CAST(N'2002-10-24T00:00:00.000' AS DateTime), 105, 29, NULL, 48, 5, 17, 99, 4, 31000000, 177394432, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (206, N'Hot Fuzz', CAST(N'2007-02-14T00:00:00.000' AS DateTime), 106, 30, NULL, 240, 1, 7, 121, 5, 12000000, 80700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (207, N'Leon', CAST(N'1994-09-14T00:00:00.000' AS DateTime), 56, 18, NULL, 79, 1, 9, 110, 6, 16000000, 45284974, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (208, N'High Plains Drifter', CAST(N'1973-08-22T00:00:00.000' AS DateTime), 10, 31, NULL, 241, 1, 1, 105, 6, 5500000, 15700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (209, N'Collateral', CAST(N'2004-08-06T00:00:00.000' AS DateTime), 34, 2, NULL, 241, 1, 9, 120, 5, 65000000, 217800000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (210, N'American Beauty', CAST(N'1999-09-08T00:00:00.000' AS DateTime), 61, 98, NULL, 241, 1, 2, 122, 6, 15000000, 356296601, 8, 5);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (211, N'Lethal Weapon', CAST(N'1987-03-06T00:00:00.000' AS DateTime), 30, 52, NULL, 241, 1, 4, 110, 6, 15000000, 120207127, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (212, N'Lethal Weapon 2', CAST(N'1989-07-07T00:00:00.000' AS DateTime), 30, 52, NULL, 241, 1, 4, 114, 5, 25000000, 227853986, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (213, N'Crouching Tiger, Hidden Dragon', CAST(N'2000-07-06T00:00:00.000' AS DateTime), 8, 99, NULL, 48, 5, 17, 120, 3, 17000000, 213525736, 10, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (214, N'Brokeback Mountain', CAST(N'2005-12-09T00:00:00.000' AS DateTime), 8, 100, NULL, 241, 1, 2, 134, 5, 14000000, 178000000, 8, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (215, N'Hulk', CAST(N'2003-06-20T00:00:00.000' AS DateTime), 8, 1, NULL, 241, 1, 4, 138, 4, 137000000, 245300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (216, N'Sense and Sensibility', CAST(N'1995-12-13T00:00:00.000' AS DateTime), 8, 5, NULL, 240, 1, 6, 136, 1, 16000000, 134993774, 7, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (217, N'Ride with the Devil', CAST(N'1999-11-26T00:00:00.000' AS DateTime), 8, 1, NULL, 241, 1, 1, 138, 5, 38000000, 635096, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (218, N'Twins', CAST(N'1988-12-09T00:00:00.000' AS DateTime), 94, 1, NULL, 241, 1, 7, 105, 2, 18000000, 216614388, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (219, N'28 Days Later', CAST(N'2002-11-01T00:00:00.000' AS DateTime), 108, 32, NULL, 240, 1, 10, 113, 6, 8000000, 82719885, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (220, N'The Beach', CAST(N'2000-02-11T00:00:00.000' AS DateTime), 108, 4, NULL, 240, 1, 2, 115, 5, 50000000, 144000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (221, N'Trainspotting', CAST(N'1996-02-23T00:00:00.000' AS DateTime), 108, 33, NULL, 240, 1, 2, 93, 6, 3100000, 72000000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (222, N'The Spy Who Loved Me', CAST(N'1977-07-07T00:00:00.000' AS DateTime), 92, 51, NULL, 240, 1, 9, 125, 2, 14000000, 185400000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (223, N'Downfall', CAST(N'2004-09-16T00:00:00.000' AS DateTime), 109, 101, NULL, 257, 4, 5, 155, 5, 13500000, 72649179, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (224, N'Letters from Iwo Jima', CAST(N'2006-12-09T00:00:00.000' AS DateTime), 10, 8, NULL, 241, 6, 5, 141, 5, 19000000, 68673228, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (225, N'Star Trek II: The Wrath of Khan', CAST(N'1982-06-04T00:00:00.000' AS DateTime), 96, 2, NULL, 241, 1, 3, 113, 3, 11200000, 97000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (226, N'House of Flying Daggers', CAST(N'2004-05-19T00:00:00.000' AS DateTime), 105, 99, NULL, 48, 5, 17, 119, 5, 12000000, 92863945, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (227, N'Das Boot', CAST(N'1981-09-17T00:00:00.000' AS DateTime), 25, 35, NULL, 257, 4, 5, 149, 5, 12000000, 84970337, 6, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (228, N'Casino Royale', CAST(N'1967-04-28T00:00:00.000' AS DateTime), 124, 5, NULL, 240, 1, 9, 131, 1, 12000000, 41700000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (229, N'Star Wars: Episode IV: A New Hope', CAST(N'1977-05-25T00:00:00.000' AS DateTime), 7, 13, NULL, 241, 1, 3, 121, 1, 11000000, 775400000, 11, 7);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (230, N'E.T.: The Extra-Terrestrial', CAST(N'1982-06-11T00:00:00.000' AS DateTime), 4, 41, NULL, 241, 1, 3, 115, 1, 10500000, 792900000, 9, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (231, N'Amelie', CAST(N'2001-04-25T00:00:00.000' AS DateTime), 111, 102, NULL, 79, 3, 7, 123, 5, 10000000, 173921954, 5, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (232, N'You Only Live Twice', CAST(N'1967-06-12T00:00:00.000' AS DateTime), 92, 51, NULL, 241, 1, 9, 117, 2, 10300000, 111000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (233, N'Thunderball', CAST(N'1965-12-09T00:00:00.000' AS DateTime), 112, 51, NULL, 240, 1, 9, 130, 2, 9000000, 141200000, 1, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (234, N'Alien', CAST(N'1979-05-25T00:00:00.000' AS DateTime), 51, 85, NULL, 241, 1, 3, 117, 6, 11000000, 203600000, 2, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (236, N'On Her Majesty''s Secret Service', CAST(N'1969-12-18T00:00:00.000' AS DateTime), 113, 51, NULL, 240, 1, 9, 140, 2, 7000000, 64600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (237, N'Pulp Fiction', CAST(N'1994-10-14T00:00:00.000' AS DateTime), 79, 84, NULL, 241, 1, 14, 154, 6, 8500000, 213900000, 7, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (238, N'Rumble in the Bronx', CAST(N'1995-01-21T00:00:00.000' AS DateTime), 114, 103, NULL, 265, 2, 17, 106, 5, 7500000, 32392047, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (239, N'Diamonds are Forever', CAST(N'1971-12-14T00:00:00.000' AS DateTime), 115, 51, NULL, 240, 1, 9, 120, 2, 7200000, 116000000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (240, N'Fargo', CAST(N'1996-03-08T00:00:00.000' AS DateTime), 5, 104, NULL, 241, 1, 14, 98, 6, 7000000, 60611975, 7, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (241, N'Live and Let Die', CAST(N'1973-06-27T00:00:00.000' AS DateTime), 115, 51, NULL, 240, 1, 9, 121, 2, 7000000, 126400000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (242, N'The Man with the Golden Gun', CAST(N'1974-12-19T00:00:00.000' AS DateTime), 115, 51, NULL, 240, 1, 9, 125, 2, 7000000, 98500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (243, N'Good Night, and Good Luck', CAST(N'2005-10-07T00:00:00.000' AS DateTime), 116, 105, NULL, 241, 1, 2, 93, 2, 7000000, 56500000, 6, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (244, N'Pale Rider', CAST(N'1985-06-26T00:00:00.000' AS DateTime), 10, 31, NULL, 241, 1, 1, 116, 5, 6900000, 41410568, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (245, N'The Terminator', CAST(N'1984-10-26T00:00:00.000' AS DateTime), 15, 106, NULL, 241, 1, 3, 108, 5, 6400000, 78400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (246, N'Around the World in 80 Days', CAST(N'1956-10-17T00:00:00.000' AS DateTime), 117, 22, NULL, 241, 1, 8, 183, 1, 6000000, 42000000, 8, 5);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (249, N'The Usual Suspects', CAST(N'1995-08-25T00:00:00.000' AS DateTime), 14, 107, NULL, 241, 1, 14, 106, 6, 6000000, 23300000, 2, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (250, N'Memento', CAST(N'2000-09-05T00:00:00.000' AS DateTime), 27, 108, NULL, 241, 1, 9, 113, 5, 9000000, 39700000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (251, N'Shaun of the Dead', CAST(N'2004-04-09T00:00:00.000' AS DateTime), 106, 30, NULL, 240, 1, 7, 99, 5, 6000000, 30000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (252, N'Four Weddings and a Funeral', CAST(N'1994-05-13T00:00:00.000' AS DateTime), 26, 33, NULL, 240, 1, 6, 117, 5, 4500000, 257700832, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (253, N'Night Watch', CAST(N'2004-07-08T00:00:00.000' AS DateTime), 119, 109, NULL, 258, 9, 12, 114, 5, 4200000, 33899078, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (254, N'Goldfinger', CAST(N'1964-09-17T00:00:00.000' AS DateTime), 115, 51, NULL, 241, 1, 9, 110, 2, 3000000, 124900000, 1, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (255, N'The Lives of Others', CAST(N'2006-03-23T00:00:00.000' AS DateTime), 120, 36, NULL, 257, 4, 2, 137, 5, 2000000, 77356942, 1, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (256, N'From Russia with Love', CAST(N'1963-10-10T00:00:00.000' AS DateTime), 112, 51, NULL, 240, 1, 9, 115, 2, 2000000, 78900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (257, N'Reservoir Dogs', CAST(N'1992-10-23T00:00:00.000' AS DateTime), 79, 110, NULL, 241, 1, 14, 99, 6, 1200000, 22000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (258, N'Dr. No', CAST(N'1962-10-05T00:00:00.000' AS DateTime), 112, 51, NULL, 240, 1, 9, 109, 2, 1100000, 59500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (259, N'King Kong', CAST(N'1933-04-07T00:00:00.000' AS DateTime), 121, 37, NULL, 241, 1, 8, 100, 2, 672000, 2847000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (260, N'Seven Samurai', CAST(N'1954-04-26T00:00:00.000' AS DateTime), 122, 38, N'I believe that this film is the origin of the phrase "don''t bring a knife to a gunfight".;
In seriousness, it''s bloody excellent.', 118, 6, 4, 207, 2, 2000000, 4500000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (261, N'Super Size Me', CAST(N'2004-09-10T00:00:00.000' AS DateTime), 123, 111, NULL, 241, 1, 24, 98, 4, 65000, 29529368, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (262, N'Kagemusha', CAST(N'1980-04-26T00:00:00.000' AS DateTime), 122, 38, NULL, 118, 6, 2, 180, 2, 7500000, 26000000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (263, N'Ran', CAST(N'1985-05-31T00:00:00.000' AS DateTime), 122, 112, NULL, 118, 6, 2, 162, 5, 12000000, 15000000, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (264, N'Sunshine', CAST(N'2007-04-06T00:00:00.000' AS DateTime), 108, 113, NULL, 240, 1, 3, 107, 5, 40000000, 32017803, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (265, N'Once Were Warriors', CAST(N'1995-04-14T00:00:00.000' AS DateTime), 29, 114, NULL, 167, 1, 2, 102, 6, 1000000, 1608570, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (266, N'Flushed Away', CAST(N'2006-11-03T00:00:00.000' AS DateTime), 125, 115, N'I saw this on TV last Christmas but I wasn''t really paying attention because I was building a Lego Super Star Destroyer at the time. It was probably ok, 7 out of 10.', 240, 1, 11, 85, 2, 149000000, 178120010, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (267, N'Indiana Jones and the Kingdom of the Crystal Skull', CAST(N'2008-05-22T00:00:00.000' AS DateTime), 4, 13, N'If I was Indiana Jones and I''d just discovered that my son was Shia LaBoeuf I''d spend the rest of the movie attempting to rediscover the Ark of the Covenant just so that I could deliberately open it directly into my own face. Which is exactly what I wanted to do after I watched this.', 241, 1, 8, 122, 4, 185000000, 786600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (270, N'Transformers: Age of Extinction', CAST(N'2014-06-27T00:00:00.000' AS DateTime), 18, 46, N'Haven''t seen it. It sounds dreadful.', 241, 1, 3, 165, 4, 210000000, 1091000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (271, N'The Hobbit: The Battle of the Five Armies', CAST(N'2014-12-01T00:00:00.000' AS DateTime), 12, 14, N'Haven''t watched it yet.', 167, 1, 12, 174, 4, 250000000, 955100000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (272, N'Guardians of the Galaxy', CAST(N'2014-08-01T00:00:00.000' AS DateTime), 128, 53, N'Probably my favourite of the Marvel universe movies. That soundtrack!', 241, 1, 3, 122, 4, 195900000, 774200000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (273, N'Maleficent', CAST(N'2014-05-28T00:00:00.000' AS DateTime), 130, 3, NULL, 241, 1, 12, 98, 2, 180000000, 758400000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (274, N'X-Men: Days of Future Past', CAST(N'2014-05-22T00:00:00.000' AS DateTime), 14, 53, NULL, 241, 1, 4, 131, 4, 200000000, 748100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (275, N'Captain America: The Winter Soldier', CAST(N'2014-04-04T00:00:00.000' AS DateTime), 132, 53, NULL, 241, 1, 4, 136, 4, 170000000, 714400000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (276, N'The Hunger Games: Mockingjay Part 1', CAST(N'2014-11-10T00:00:00.000' AS DateTime), 23, 120, NULL, 241, 1, 8, 123, 4, 125000000, 755400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (277, N'The Amazing Spider-Man 2', CAST(N'2014-04-10T00:00:00.000' AS DateTime), 135, 53, NULL, 241, 1, 4, 141, 4, 255000000, 709000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (278, N'Dawn of the Planet of the Apes', CAST(N'2014-07-11T00:00:00.000' AS DateTime), 137, 123, NULL, 241, 1, 4, 130, 4, 170000000, 710600000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (279, N'Interstellar', CAST(N'2014-11-07T00:00:00.000' AS DateTime), 27, 43, NULL, 241, 1, 3, 169, 4, 165000000, 675100000, 5, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (280, N'Frozen', CAST(N'2013-11-27T00:00:00.000' AS DateTime), 139, 3, N'I''ve never watched it but if I hear that song one more time...', 241, 1, 11, 102, 2, 150000000, 1276000000, 2, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (281, N'Iron Man 3', CAST(N'2013-05-03T00:00:00.000' AS DateTime), 140, 53, NULL, 241, 1, 4, 130, 4, 200000000, 1215000000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (282, N'Despicable Me 2', CAST(N'2013-07-03T00:00:00.000' AS DateTime), 141, 124, NULL, 241, 1, 11, 98, 1, 76000000, 970800000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (283, N'The Hobbit: The Desolation of Smaug', CAST(N'2013-12-13T00:00:00.000' AS DateTime), 12, 42, NULL, 167, 1, 12, 161, 4, 225000000, 958400000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (284, N'The Hunger Games: Catching Fire', CAST(N'2013-11-11T00:00:00.000' AS DateTime), 23, 120, NULL, 241, 1, 8, 146, 4, 130000000, 865000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (285, N'Fast & Furious 6', CAST(N'2013-05-07T00:00:00.000' AS DateTime), 142, 54, NULL, 241, 1, 4, 130, 4, 160000000, 788700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (286, N'Monsters University', CAST(N'2013-06-21T00:00:00.000' AS DateTime), 143, 10, NULL, 241, 1, 11, 104, 1, 200000000, 743600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (287, N'Gravity', CAST(N'2013-10-04T00:00:00.000' AS DateTime), 38, 47, NULL, 241, 1, 3, 91, 4, 100000000, 723200000, 10, 7);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (288, N'Man of Steel', CAST(N'2013-06-14T00:00:00.000' AS DateTime), 76, 43, NULL, 241, 1, 4, 143, 4, 225000000, 668000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (289, N'Thor: The Dark World', CAST(N'2013-10-22T00:00:00.000' AS DateTime), 144, 53, NULL, 241, 1, 4, 112, 4, 170000000, 644600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (290, N'The Avengers', CAST(N'2012-05-04T00:00:00.000' AS DateTime), 86, 53, NULL, 241, 1, 4, 143, 4, 220000000, 1520000000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (291, N'Skyfall', CAST(N'2012-10-26T00:00:00.000' AS DateTime), 61, 51, NULL, 240, 1, 9, 143, 4, 150000000, 1109000000, 5, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (292, N'The Dark Knight Rises', CAST(N'2012-07-20T00:00:00.000' AS DateTime), 27, 43, NULL, 241, 1, 4, 165, 4, 230000000, 1085000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (293, N'The Hobbit: An Unexpected Journey', CAST(N'2012-12-14T00:00:00.000' AS DateTime), 12, 42, NULL, 167, 1, 12, 169, 4, 200000000, 1021000000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (294, N'Ice Age: Continental Drift', CAST(N'2012-06-27T00:00:00.000' AS DateTime), 149, 4, NULL, 241, 1, 11, 88, 1, 95000000, 877200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (295, N'The Twilight Saga: Breaking Dawn - Part 2', CAST(N'2012-11-16T00:00:00.000' AS DateTime), 150, 108, N'It''s over! It''s finally over!', 241, 1, 28, 115, 4, 120000000, 829700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (296, N'The Amazing Spider-Man', CAST(N'2012-07-03T00:00:00.000' AS DateTime), 135, 53, NULL, 241, 1, 4, 136, 4, 230000000, 757900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (297, N'Madagascar 3: Europe''s Most Wanted', CAST(N'2012-06-08T00:00:00.000' AS DateTime), 151, 8, NULL, 241, 1, 11, 93, 2, 145000000, 746900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (298, N'The Hunger Games', CAST(N'2012-03-23T00:00:00.000' AS DateTime), 152, 120, NULL, 241, 1, 3, 142, 4, 78000000, 694400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (299, N'Men in Black 3', CAST(N'2012-05-25T00:00:00.000' AS DateTime), 31, 41, NULL, 241, 1, 3, 106, 2, 215000000, 624000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (300, N'Harry Potter and the Deathly Hallows - Part 2', CAST(N'2011-07-15T00:00:00.000' AS DateTime), 19, 47, NULL, 240, 1, 12, 130, 4, 250000000, 1342000000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (301, N'Transformers: Dark of the Moon', CAST(N'2011-06-29T00:00:00.000' AS DateTime), 18, 46, NULL, 241, 1, 3, 157, 4, 195000000, 1124000000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (302, N'Pirates of the Caribbean: On Stranger Tides', CAST(N'2011-05-20T00:00:00.000' AS DateTime), 153, 3, NULL, 241, 1, 8, 137, 4, 378500000, 1046000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (303, N'The Twilight Saga: Breaking Dawn - Part 1', CAST(N'2011-11-18T00:00:00.000' AS DateTime), 150, 125, N'They actually made an extended cut of this one. Why won''t it just end? :(', 241, 1, 28, 117, 4, 110000000, 712200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (304, N'Mission: Impossible - Ghost Protocol', CAST(N'2011-12-16T00:00:00.000' AS DateTime), 24, 126, NULL, 241, 1, 9, 133, 4, 145000000, 694700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (305, N'Kung Fu Panda 2', CAST(N'2011-05-26T00:00:00.000' AS DateTime), 154, 8, NULL, 241, 1, 11, 90, 2, 150000000, 665700000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (306, N'Fast Five', CAST(N'2011-04-29T00:00:00.000' AS DateTime), 142, 54, NULL, 241, 1, 4, 130, 4, 125000000, 626100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (307, N'The Hangover Part II', CAST(N'2011-05-26T00:00:00.000' AS DateTime), 75, 43, NULL, 241, 1, 7, 102, 5, 80000000, 586800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (308, N'The Smurfs', CAST(N'2011-07-29T00:00:00.000' AS DateTime), 155, 127, NULL, 241, 1, 11, 103, 1, 110000000, 563700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (309, N'Cars 2', CAST(N'2011-06-24T00:00:00.000' AS DateTime), 156, 10, NULL, 241, 1, 11, 106, 1, 200000000, 559900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (310, N'Toy Story 3', CAST(N'2010-06-18T00:00:00.000' AS DateTime), 157, 10, NULL, 241, 1, 11, 103, 1, 200000000, 1063000000, 5, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (311, N'Alice in Wonderland', CAST(N'2010-02-25T00:00:00.000' AS DateTime), 28, 3, NULL, 240, 1, 12, 108, 2, 150000000, 1025000000, 3, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (312, N'Harry Potter and the Deathly Hallows - Part 1', CAST(N'2010-11-19T00:00:00.000' AS DateTime), 19, 47, NULL, 240, 1, 12, 146, 4, 250000000, 960300000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (313, N'Inception', CAST(N'2010-07-16T00:00:00.000' AS DateTime), 27, 43, NULL, 241, 1, 3, 148, 4, 160000000, 825500000, 8, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (314, N'Shrek Forever After', CAST(N'2010-05-21T00:00:00.000' AS DateTime), 158, 8, NULL, 241, 1, 11, 93, 1, 135000000, 753000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (315, N'The Twilight Saga: Eclipse', CAST(N'2010-06-30T00:00:00.000' AS DateTime), 159, 125, N'Another one?! Ergh. Just... No.', 241, 1, 28, 123, 4, 68000000, 698500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (316, N'Iron Man 2', CAST(N'2010-05-07T00:00:00.000' AS DateTime), 160, 53, NULL, 241, 1, 4, 125, 4, 200000000, 623900000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (317, N'Tangled', CAST(N'2010-11-24T00:00:00.000' AS DateTime), 161, 3, NULL, 241, 1, 11, 100, 2, 260000000, 591800000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (318, N'Despicable Me', CAST(N'2010-07-09T00:00:00.000' AS DateTime), 141, 124, NULL, 241, 1, 11, 95, 1, 69000000, 543200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (319, N'How to Train Your Dragon', CAST(N'2010-03-26T00:00:00.000' AS DateTime), 162, 8, NULL, 241, 1, 11, 98, 2, 165000000, 494900000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (320, N'Avatar', CAST(N'2009-12-17T00:00:00.000' AS DateTime), 15, 70, N'Dances With Wolves in space.', 241, 1, 3, 161, 4, 237000000, 2788000000, 9, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (321, N'Harry Potter and the Half-Blood Prince', CAST(N'2009-07-15T00:00:00.000' AS DateTime), 19, 47, NULL, 240, 1, 12, 153, 4, 250000000, 934400000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (322, N'Ice Age: Dawn of the Dinosaurs', CAST(N'2009-07-01T00:00:00.000' AS DateTime), 163, 4, NULL, 241, 1, 11, 94, 1, 90000000, 886700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (323, N'Transformers: Revenge of the Fallen', CAST(N'2009-06-19T00:00:00.000' AS DateTime), 18, 46, NULL, 241, 1, 3, 150, 4, 200000000, 836300000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (324, N'2012', CAST(N'2009-11-13T00:00:00.000' AS DateTime), 42, 57, N'Danny Glover was way tool old for this sh*t.', 241, 1, 25, 158, 4, 200000000, 769700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (325, N'Up', CAST(N'2009-05-29T00:00:00.000' AS DateTime), 46, 10, N'I totally didn''t cry within the first five minutes of this film. It was just some dust in my eye. Oh, hey look, a squirrel!', 241, 1, 11, 96, 1, 175000000, 731400000, 5, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (326, N'The Twilight Saga: New Moon', CAST(N'2009-11-20T00:00:00.000' AS DateTime), 164, 125, N'You know what would make these films infinitely better? Wesley Snipes as Blade.', 241, 1, 28, 130, 4, 50000000, 709700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (327, N'Sherlock Holmes', CAST(N'2009-12-26T00:00:00.000' AS DateTime), 165, 52, NULL, 240, 1, 18, 128, 4, 90000000, 524000000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (328, N'Angels & Demons', CAST(N'2009-05-15T00:00:00.000' AS DateTime), 40, 20, NULL, 241, 1, 18, 138, 5, 150000000, 485900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (329, N'The Hangover', CAST(N'2009-06-05T00:00:00.000' AS DateTime), 75, 43, NULL, 241, 1, 7, 100, 5, 35000000, 467500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (330, N'The Dark Knight', CAST(N'2008-07-24T00:00:00.000' AS DateTime), 27, 43, NULL, 241, 1, 4, 152, 4, 185000000, 1005000000, 8, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (332, N'Kung Fu Panda', CAST(N'2008-06-06T00:00:00.000' AS DateTime), 166, 8, NULL, 241, 1, 11, 92, 2, 130000000, 631700000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (333, N'Hancock', CAST(N'2008-07-02T00:00:00.000' AS DateTime), 167, 128, NULL, 241, 1, 4, 92, 4, 150000000, 624300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (334, N'Mamma Mia!', CAST(N'2008-06-30T00:00:00.000' AS DateTime), 168, 129, NULL, 240, 1, 20, 109, 2, 52000000, 609800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (335, N'Madagascar: Escape 2 Africa', CAST(N'2008-11-07T00:00:00.000' AS DateTime), 151, 8, NULL, 241, 1, 11, 89, 2, 150000000, 602300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (336, N'Quantum of Solace', CAST(N'2008-10-31T00:00:00.000' AS DateTime), 169, 51, NULL, 240, 1, 9, 106, 4, 200000000, 586100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (337, N'Iron Man', CAST(N'2008-05-02T00:00:00.000' AS DateTime), 160, 53, NULL, 241, 1, 4, 126, 4, 140000000, 585200000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (338, N'WALL-E', CAST(N'2008-06-27T00:00:00.000' AS DateTime), 170, 10, NULL, 241, 1, 11, 98, 1, 180000000, 521300000, 6, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (339, N'The Chronicles of Narnia: Prince Caspian', CAST(N'2008-05-16T00:00:00.000' AS DateTime), 67, 3, NULL, 240, 1, 12, 150, 2, 225000000, 419700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (340, N'National Treasure: Book of Secrets', CAST(N'2007-12-21T00:00:00.000' AS DateTime), 171, 17, NULL, 241, 1, 8, 124, 2, 130000000, 457400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (341, N'Frost/Nixon', CAST(N'2008-10-15T00:00:00.000' AS DateTime), 40, 20, NULL, 241, 1, 2, 122, 5, 25000000, 27400000, 5, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (342, N'Ice Age: The Meltdown', CAST(N'2006-03-31T00:00:00.000' AS DateTime), 163, 4, NULL, 241, 1, 11, 91, 1, 80000000, 660900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (343, N'Night at the Museum', CAST(N'2006-12-22T00:00:00.000' AS DateTime), 172, 28, NULL, 241, 1, 8, 108, 2, 110000000, 574500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (344, N'Cars', CAST(N'2006-06-09T00:00:00.000' AS DateTime), 156, 10, NULL, 241, 1, 11, 116, 2, 120000000, 462000000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (345, N'Happy Feet', CAST(N'2006-11-17T00:00:00.000' AS DateTime), 173, 49, NULL, 261, 1, 11, 108, 1, 100000000, 384300000, 1, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (346, N'The Chronicles of Narnia: The Lion, the Witch and the Wardrobe', CAST(N'2005-12-08T00:00:00.000' AS DateTime), 67, 63, NULL, 240, 1, 12, 145, 2, 180000000, 745000000, 3, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (347, N'Madagascar', CAST(N'2005-05-27T00:00:00.000' AS DateTime), 151, 8, NULL, 241, 1, 11, 86, 1, 75000000, 532700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (348, N'Hitch', CAST(N'2005-02-11T00:00:00.000' AS DateTime), 174, 130, NULL, 241, 1, 6, 118, 4, 70000000, 368100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (349, N'The Incredibles', CAST(N'2004-11-05T00:00:00.000' AS DateTime), 24, 10, N'Probably the best superhero movie ever.', 241, 1, 11, 115, 1, 92000000, 633000000, 4, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (350, N'Meet the Fockers', CAST(N'2004-12-22T00:00:00.000' AS DateTime), 175, 131, NULL, 241, 1, 7, 115, 4, 80000000, 516000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (351, N'Shark Tale', CAST(N'2004-10-01T00:00:00.000' AS DateTime), 176, 8, NULL, 241, 1, 11, 90, 1, 75000000, 367300000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (352, N'Finding Nemo', CAST(N'2003-05-30T00:00:00.000' AS DateTime), 170, 10, N'Spoiler: they find him.', 241, 1, 11, 100, 1, 94000000, 940300000, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (353, N'Terminator 3: Rise of the Machines', CAST(N'2003-07-02T00:00:00.000' AS DateTime), 177, 132, NULL, 241, 1, 3, 109, 4, 187300000, 433400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (354, N'Bad Boys II', CAST(N'2003-07-18T00:00:00.000' AS DateTime), 18, 17, NULL, 241, 1, 4, 147, 5, 130000000, 273300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (355, N'Signs', CAST(N'2002-08-02T00:00:00.000' AS DateTime), 62, 7, NULL, 241, 1, 18, 107, 4, 72000000, 408200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (356, N'Ice Age', CAST(N'2002-03-15T00:00:00.000' AS DateTime), 163, 4, NULL, 241, 1, 11, 81, 1, 59000000, 383300000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (357, N'My Big Fat Greek Wedding', CAST(N'2002-04-19T00:00:00.000' AS DateTime), 184, 135, NULL, 262, 1, 7, 95, 2, 5000000, 368700000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (358, N'The Mummy Returns', CAST(N'2001-05-04T00:00:00.000' AS DateTime), 186, 137, NULL, 241, 1, 8, 130, 3, 98000000, 433000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (359, N'Jurassic Park III', CAST(N'2001-07-18T00:00:00.000' AS DateTime), 187, 41, NULL, 241, 1, 8, 92, 2, 93000000, 368800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (360, N'Hannibal', CAST(N'2001-02-09T00:00:00.000' AS DateTime), 51, 96, NULL, 241, 1, 9, 132, 6, 87000000, 351600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (361, N'What Women Want', CAST(N'2000-12-15T00:00:00.000' AS DateTime), 188, 25, NULL, 241, 1, 7, 127, 3, 70000000, 374111707, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (362, N'Dinosaur', CAST(N'2000-05-19T00:00:00.000' AS DateTime), 189, 3, NULL, 241, 1, 11, 82, 2, 127500000, 349800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (363, N'How the Grinch Stole Christmas', CAST(N'2000-11-17T00:00:00.000' AS DateTime), 40, 20, NULL, 241, 1, 7, 104, 2, 123000000, 345100000, 3, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (364, N'Meet the Parents', CAST(N'2000-10-06T00:00:00.000' AS DateTime), 175, 131, NULL, 241, 1, 7, 108, 3, 55000000, 330400000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (365, N'What Lies Beneath', CAST(N'2000-07-21T00:00:00.000' AS DateTime), 20, 9, NULL, 241, 1, 10, 130, 5, 100000000, 291400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (366, N'Toy Story 2', CAST(N'1999-11-24T00:00:00.000' AS DateTime), 156, 10, NULL, 241, 1, 11, 95, 1, 90000000, 497400000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (367, N'Tarzan', CAST(N'1999-06-16T00:00:00.000' AS DateTime), 190, 3, NULL, 241, 1, 11, 88, 1, 130000000, 448200000, 1, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (368, N'The Mummy', CAST(N'1999-05-07T00:00:00.000' AS DateTime), 186, 137, NULL, 241, 1, 8, 125, 3, 80000000, 415900000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (369, N'Notting Hill', CAST(N'1999-05-21T00:00:00.000' AS DateTime), 191, 104, NULL, 240, 1, 27, 124, 5, 43000000, 363889678, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (370, N'Austin Powers: The Spy Who Shagged Me', CAST(N'1999-06-11T00:00:00.000' AS DateTime), 175, 138, NULL, 241, 1, 7, 96, 3, 33000000, 312000000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (371, N'Godzilla', CAST(N'1998-05-20T00:00:00.000' AS DateTime), 42, 57, NULL, 241, 1, 3, 139, 2, 130000000, 379000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (372, N'There''s Something About Mary', CAST(N'1998-07-15T00:00:00.000' AS DateTime), 192, 4, NULL, 241, 1, 7, 119, 5, 23000000, 369900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (373, N'A Bug''s Life', CAST(N'1998-11-25T00:00:00.000' AS DateTime), 156, 10, NULL, 241, 1, 11, 95, 1, 120000000, 363300000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (374, N'Mulan', CAST(N'1998-06-19T00:00:00.000' AS DateTime), 193, 3, NULL, 241, 1, 11, 87, 1, 90000000, 304300000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (375, N'Dr. Dolittle', CAST(N'1998-06-26T00:00:00.000' AS DateTime), 194, 68, NULL, 241, 1, 7, 85, 2, 70500000, 294400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (376, N'Shakespeare in Love', CAST(N'1998-12-03T00:00:00.000' AS DateTime), 195, 69, NULL, 240, 1, 27, 123, 5, 25000000, 289300000, 13, 7);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (377, N'The Lost World: Jurassic Park', CAST(N'1997-05-23T00:00:00.000' AS DateTime), 4, 41, NULL, 241, 1, 8, 129, 2, 73000000, 618600000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (378, N'Air Force One', CAST(N'1997-07-25T00:00:00.000' AS DateTime), 25, 7, NULL, 241, 1, 9, 124, 5, 85000000, 315100000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (380, N'Liar Liar', CAST(N'1997-03-21T00:00:00.000' AS DateTime), 16, 20, NULL, 241, 1, 7, 88, 3, 45000000, 302700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (381, N'My Best Friend''s Wedding', CAST(N'1997-06-20T00:00:00.000' AS DateTime), 196, 139, NULL, 241, 1, 27, 104, 3, 38000000, 299300000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (382, N'The Full Monty', CAST(N'1997-08-29T00:00:00.000' AS DateTime), 197, 33, NULL, 240, 1, 7, 91, 5, 3500000, 258000000, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (383, N'Twister', CAST(N'1996-05-17T00:00:00.000' AS DateTime), 198, 41, NULL, 241, 1, 25, 113, 2, 92000000, 494400000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (384, N'The Hunchback of Notre Dame', CAST(N'1996-06-21T00:00:00.000' AS DateTime), 199, 3, NULL, 241, 1, 11, 91, 1, 100000000, 325300000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (385, N'101 Dalmatians', CAST(N'1996-11-27T00:00:00.000' AS DateTime), 200, 3, NULL, 241, 1, 26, 103, 1, 75000000, 320600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (386, N'Ransom', CAST(N'1996-11-08T00:00:00.000' AS DateTime), 40, 7, NULL, 241, 1, 9, 121, 5, 80000000, 309500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (387, N'The Nutty Professor', CAST(N'1996-06-28T00:00:00.000' AS DateTime), 16, 20, NULL, 241, 1, 7, 95, 3, 54000000, 274000000, 1, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (388, N'Jerry Maguire', CAST(N'1996-12-13T00:00:00.000' AS DateTime), 201, 77, NULL, 241, 1, 2, 139, 5, 50000000, 273600000, 5, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (389, N'Eraser', CAST(N'1996-06-21T00:00:00.000' AS DateTime), 202, 140, NULL, 241, 1, 4, 114, 5, 100000000, 242300000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (390, N'Toy Story', CAST(N'1995-11-22T00:00:00.000' AS DateTime), 156, 10, NULL, 241, 1, 11, 81, 2, 30000000, 373600000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (391, N'Pocahontas', CAST(N'1995-06-23T00:00:00.000' AS DateTime), 203, 3, NULL, 241, 1, 11, 82, 1, 55000000, 346100000, 2, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (392, N'Batman Forever', CAST(N'1995-06-16T00:00:00.000' AS DateTime), 204, 104, NULL, 241, 1, 4, 122, 3, 100000000, 336500000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (393, N'Casper', CAST(N'1995-05-26T00:00:00.000' AS DateTime), 205, 41, NULL, 241, 1, 7, 101, 2, 55000000, 287900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (394, N'Jumanji', CAST(N'1995-12-15T00:00:00.000' AS DateTime), 187, 141, NULL, 241, 1, 8, 104, 2, 65000000, 262800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (395, N'The Lion King', CAST(N'1994-06-15T00:00:00.000' AS DateTime), 379, 3, NULL, 241, 1, 11, 88, 1, 45000000, 968500000, 4, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (396, N'The Mask', CAST(N'1994-07-29T00:00:00.000' AS DateTime), 202, 142, NULL, 241, 1, 7, 101, 2, 23000000, 351600000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (397, N'Speed', CAST(N'1994-06-10T00:00:00.000' AS DateTime), 198, 143, NULL, 241, 1, 4, 115, 5, 30000000, 350400000, 3, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (398, N'The Flintstones', CAST(N'1994-05-27T00:00:00.000' AS DateTime), 208, 41, NULL, 241, 1, 7, 91, 1, 46000000, 341600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (399, N'Dumb and Dumber', CAST(N'1994-12-16T00:00:00.000' AS DateTime), 192, 144, NULL, 241, 1, 7, 107, 3, 17000000, 247000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (400, N'Donnie Brasco', CAST(N'1997-02-28T00:00:00.000' AS DateTime), 26, 33, NULL, 241, 1, 14, 126, 6, 35000000, 124900000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (401, N'Interview with the Vampire: The Vampire Chronicles', CAST(N'1994-11-11T00:00:00.000' AS DateTime), 209, 145, NULL, 241, 1, 10, 122, 6, 60000000, 223700000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (403, N'Mrs. Doubtfire', CAST(N'1993-11-24T00:00:00.000' AS DateTime), 41, 146, NULL, 241, 1, 7, 125, 3, 25000000, 441300000, 1, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (404, N'The Firm', CAST(N'1993-06-30T00:00:00.000' AS DateTime), 210, 68, NULL, 241, 1, 9, 154, 5, 42000000, 270248367, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (405, N'Indecent Proposal', CAST(N'1993-04-07T00:00:00.000' AS DateTime), 211, 2, NULL, 241, 1, 2, 118, 5, 38000000, 266600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (406, N'Sleepless in Seattle', CAST(N'1993-06-25T00:00:00.000' AS DateTime), 212, 147, NULL, 241, 1, 6, 105, 2, 21000000, 227800000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (407, N'Philadelphia', CAST(N'1993-12-22T00:00:00.000' AS DateTime), 213, 147, NULL, 241, 1, 2, 126, 3, 26000000, 206700000, 5, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (408, N'The Pelican Brief', CAST(N'1993-12-17T00:00:00.000' AS DateTime), 214, 6, NULL, 241, 1, 9, 141, 3, 45000000, 195300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (409, N'Aladdin', CAST(N'1992-11-25T00:00:00.000' AS DateTime), 215, 3, NULL, 241, 1, 11, 90, 1, 28000000, 504100000, 5, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (410, N'The Bodyguard', CAST(N'1992-11-25T00:00:00.000' AS DateTime), 216, 27, NULL, 241, 1, 9, 129, 5, 25000000, 411000000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (411, N'Home Alone 2: Lost in New York', CAST(N'1992-11-20T00:00:00.000' AS DateTime), 41, 148, NULL, 241, 1, 26, 120, 2, 20000000, 359000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (412, N'Basic Instinct', CAST(N'1992-05-08T00:00:00.000' AS DateTime), 72, 15, NULL, 241, 1, 9, 128, 6, 49000000, 352900000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (413, N'Batman Returns', CAST(N'1992-06-19T00:00:00.000' AS DateTime), 28, 104, NULL, 241, 1, 4, 126, 2, 80000000, 266800000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (414, N'A Few Good Men', CAST(N'1992-12-11T00:00:00.000' AS DateTime), 217, 149, NULL, 241, 1, 2, 138, 5, 40000000, 243200000, 4, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (415, N'Sister Act', CAST(N'1992-05-29T00:00:00.000' AS DateTime), 218, 7, NULL, 241, 1, 7, 100, 2, 31000000, 231600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (416, N'Bram Stoker''s Dracula', CAST(N'1992-11-13T00:00:00.000' AS DateTime), 219, 5, NULL, 241, 1, 10, 128, 6, 40000000, 215900000, 4, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (417, N'Wayne''s World', CAST(N'1992-02-14T00:00:00.000' AS DateTime), 220, 2, NULL, 241, 1, 7, 95, 2, 20000000, 183100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (418, N'Beauty and the Beast', CAST(N'1991-11-22T00:00:00.000' AS DateTime), 199, 3, NULL, 241, 1, 11, 84, 1, 25000000, 425000000, 6, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (419, N'Hook', CAST(N'1991-12-11T00:00:00.000' AS DateTime), 4, 41, NULL, 241, 1, 8, 144, 1, 70000000, 300900000, 5, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (420, N'The Silence of the Lambs', CAST(N'1991-02-14T00:00:00.000' AS DateTime), 213, 150, NULL, 241, 1, 9, 118, 6, 19000000, 272700000, 7, 5);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (421, N'JFK', CAST(N'1991-12-20T00:00:00.000' AS DateTime), 221, 65, NULL, 241, 1, 9, 188, 5, 40000000, 205400000, 8, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (422, N'The Addams Family', CAST(N'1991-11-22T00:00:00.000' AS DateTime), 31, 151, NULL, 241, 1, 7, 99, 2, 30000000, 191500000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (423, N'Cape Fear', CAST(N'1991-11-13T00:00:00.000' AS DateTime), 9, 41, NULL, 241, 1, 9, 128, 6, 35000000, 182300000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (424, N'Hot Shots!', CAST(N'1991-07-31T00:00:00.000' AS DateTime), 222, 4, NULL, 241, 1, 7, 84, 3, 26000000, 181100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (425, N'City Slickers', CAST(N'1991-06-07T00:00:00.000' AS DateTime), 223, 149, NULL, 241, 1, 7, 112, 3, 26000000, 179033791, 1, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (426, N'Ghost', CAST(N'1990-07-13T00:00:00.000' AS DateTime), 224, 2, NULL, 241, 1, 6, 127, 3, 22000000, 505700000, 5, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (427, N'Home Alone', CAST(N'1990-11-16T00:00:00.000' AS DateTime), 41, 148, NULL, 241, 1, 7, 102, 2, 18000000, 476700000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (428, N'Pretty Woman', CAST(N'1990-03-23T00:00:00.000' AS DateTime), 225, 7, NULL, 241, 1, 6, 119, 5, 14000000, 463400000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (430, N'Teenage Mutant Ninja Turtles', CAST(N'1990-03-30T00:00:00.000' AS DateTime), 226, 103, NULL, 241, 1, 4, 93, 2, 13500000, 201900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (431, N'Batman', CAST(N'1989-06-23T00:00:00.000' AS DateTime), 28, 104, NULL, 241, 1, 4, 126, 3, 35000000, 411300000, 1, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (432, N'Look Who''s Talking', CAST(N'1989-10-13T00:00:00.000' AS DateTime), 227, 147, NULL, 241, 1, 7, 96, 3, 7500000, 297000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (433, N'Dead Poets Society', CAST(N'1989-06-02T00:00:00.000' AS DateTime), 36, 7, NULL, 241, 1, 2, 128, 2, 16400000, 235800000, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (434, N'Honey, I Shrunk the Kids', CAST(N'1989-06-23T00:00:00.000' AS DateTime), 187, 3, NULL, 241, 1, 8, 94, 1, 18000000, 222700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (435, N'Ghostbusters II', CAST(N'1989-06-16T00:00:00.000' AS DateTime), 94, 5, NULL, 241, 1, 7, 108, 2, 37000000, 215400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (436, N'The Little Mermaid', CAST(N'1989-11-17T00:00:00.000' AS DateTime), 215, 3, NULL, 241, 1, 11, 82, 1, 40000000, 211300000, 3, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (437, N'Born on the Fourth of July', CAST(N'1989-12-20T00:00:00.000' AS DateTime), 221, 1, NULL, 241, 1, 19, 145, 6, 14000000, 161000000, 8, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (438, N'Rain Man', CAST(N'1988-12-16T00:00:00.000' AS DateTime), 228, 22, NULL, 241, 1, 2, 133, 5, 25000000, 354800000, 8, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (439, N'Coming to America', CAST(N'1988-06-29T00:00:00.000' AS DateTime), 229, 152, NULL, 241, 1, 7, 117, 5, 39000000, 288752301, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (440, N'Big', CAST(N'1988-12-03T00:00:00.000' AS DateTime), 230, 77, NULL, 241, 1, 7, 104, 3, 18000000, 151700000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (441, N'Crocodile Dundee II', CAST(N'1988-05-25T00:00:00.000' AS DateTime), 231, 2, NULL, 261, 1, 8, 112, 2, 14000000, 239600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (442, N'The Naked Gun: From the Files of Police Squad!', CAST(N'1988-12-02T00:00:00.000' AS DateTime), 232, 2, NULL, 241, 1, 7, 85, 5, 12000000, 78800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (443, N'Cocktail', CAST(N'1988-07-29T00:00:00.000' AS DateTime), 233, 7, NULL, 241, 1, 6, 103, 5, 20000000, 171500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (444, N'Beetlejuice', CAST(N'1988-03-30T00:00:00.000' AS DateTime), 28, 145, NULL, 241, 1, 7, 92, 5, 15000000, 73700000, 1, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (445, N'Three Men and a Baby', CAST(N'1987-11-25T00:00:00.000' AS DateTime), 99, 7, NULL, 241, 1, 7, 104, 2, 16000000, 168800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (446, N'Fatal Attraction', CAST(N'1987-09-18T00:00:00.000' AS DateTime), 211, 2, NULL, 241, 1, 9, 119, 6, 14000000, 320100000, 6, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (447, N'Beverly Hills Cop II', CAST(N'1987-05-20T00:00:00.000' AS DateTime), 78, 2, NULL, 241, 1, 4, 103, 5, 20000000, 300000000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (448, N'Good Morning, Vietnam', CAST(N'1987-12-23T00:00:00.000' AS DateTime), 228, 7, NULL, 241, 1, 7, 120, 5, 13000000, 123900000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (449, N'Moonstruck', CAST(N'1987-12-18T00:00:00.000' AS DateTime), 234, 11, NULL, 241, 1, 6, 102, 2, 15000000, 91640528, 6, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (450, N'The Secret of My Success', CAST(N'1987-04-10T00:00:00.000' AS DateTime), 235, 153, NULL, 241, 1, 7, 111, 2, 18000000, 110996879, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (451, N'Stakeout', CAST(N'1987-08-05T00:00:00.000' AS DateTime), 236, 7, NULL, 241, 1, 7, 117, 5, 14500000, 65673233, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (452, N'The Witches of Eastwick', CAST(N'1987-06-12T00:00:00.000' AS DateTime), 173, 154, NULL, 241, 1, 7, 118, 6, 22000000, 63800000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (453, N'Top Gun', CAST(N'1986-05-16T00:00:00.000' AS DateTime), 78, 2, NULL, 241, 1, 4, 110, 5, 15000000, 356800000, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (454, N'Crocodile Dundee', CAST(N'1986-09-26T00:00:00.000' AS DateTime), 237, 155, NULL, 261, 1, 7, 98, 5, 8800000, 328000000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (455, N'Platoon', CAST(N'1986-12-19T00:00:00.000' AS DateTime), 221, 106, NULL, 241, 1, 5, 120, 5, 6000000, 138500000, 8, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (456, N'The Karate Kid, Part II', CAST(N'1986-06-20T00:00:00.000' AS DateTime), 238, 5, NULL, 241, 1, 2, 113, 2, 13000000, 115100000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (458, N'The Golden Child', CAST(N'1986-12-12T00:00:00.000' AS DateTime), 239, 152, NULL, 241, 1, 8, 94, 2, 25000000, 79800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (460, N'Ferris Bueller''s Day Off', CAST(N'1986-06-11T00:00:00.000' AS DateTime), 240, 2, NULL, 241, 1, 7, 103, 5, 5800000, 70100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (461, N'Rambo: First Blood Part II', CAST(N'1985-05-22T00:00:00.000' AS DateTime), 241, 15, NULL, 241, 1, 4, 96, 5, 25500000, 300400000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (462, N'Rocky IV', CAST(N'1985-11-27T00:00:00.000' AS DateTime), 242, 22, NULL, 241, 1, 22, 91, 2, 28000000, 300400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (463, N'The Color Purple', CAST(N'1985-12-18T00:00:00.000' AS DateTime), 4, 41, NULL, 241, 1, 2, 153, 5, 15000000, 142000000, 11, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (464, N'Out Of Africa', CAST(N'1985-12-18T00:00:00.000' AS DateTime), 210, 156, NULL, 241, 1, 6, 161, 2, 28000000, 128500000, 11, 7);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (465, N'Cocoon', CAST(N'1985-06-21T00:00:00.000' AS DateTime), 40, 4, NULL, 241, 1, 3, 117, 2, 17500000, 85300000, 2, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (466, N'The Jewel of the Nile', CAST(N'1985-12-11T00:00:00.000' AS DateTime), 243, 4, NULL, 241, 1, 8, 107, 2, 25000000, 96700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (468, N'The Goonies', CAST(N'1985-06-07T00:00:00.000' AS DateTime), 30, 41, NULL, 241, 1, 8, 114, 2, 19000000, 61500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (469, N'Spies Like Us', CAST(N'1985-12-06T00:00:00.000' AS DateTime), 229, 6, NULL, 241, 1, 7, 102, 2, 22000000, 60000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (470, N'Beverly Hills Cop', CAST(N'1984-12-05T00:00:00.000' AS DateTime), 244, 152, NULL, 241, 1, 4, 105, 5, 15000000, 316400000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (472, N'Gremlins', CAST(N'1984-06-08T00:00:00.000' AS DateTime), 245, 41, NULL, 241, 1, 7, 106, 5, 11000000, 153100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (473, N'The Karate Kid', CAST(N'1984-06-22T00:00:00.000' AS DateTime), 238, 5, NULL, 241, 1, 2, 127, 2, 8000000, 90800000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (474, N'Police Academy', CAST(N'1984-03-23T00:00:00.000' AS DateTime), 246, 94, NULL, 241, 1, 7, 96, 5, 4500000, 146000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (475, N'Footloose', CAST(N'1984-02-17T00:00:00.000' AS DateTime), 235, 2, NULL, 241, 1, 2, 110, 5, 8200000, 80000000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (476, N'Romancing the Stone', CAST(N'1984-03-30T00:00:00.000' AS DateTime), 20, 4, NULL, 241, 1, 8, 106, 3, 10000000, 86500000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (477, N'Terms of Endearment', CAST(N'1983-12-09T00:00:00.000' AS DateTime), 247, 2, NULL, 241, 1, 2, 132, 5, 8000000, 108400000, 11, 5);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (478, N'Flashdance', CAST(N'1983-04-15T00:00:00.000' AS DateTime), 211, 104, NULL, 241, 1, 2, 98, 5, 7000000, 201500000, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (479, N'Trading Places', CAST(N'1983-06-10T00:00:00.000' AS DateTime), 229, 152, NULL, 241, 1, 7, 116, 5, 15000000, 90400000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (480, N'WarGames', CAST(N'1983-06-03T00:00:00.000' AS DateTime), 236, 22, NULL, 241, 1, 9, 114, 2, 12000000, 79600000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (481, N'Sudden Impact', CAST(N'1983-12-09T00:00:00.000' AS DateTime), 10, 6, NULL, 241, 1, 9, 117, 6, 22000000, 67642693, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (482, N'Staying Alive', CAST(N'1983-07-15T00:00:00.000' AS DateTime), 242, 2, NULL, 241, 1, 2, 93, 2, 22000000, 64800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (485, N'Tootsie', CAST(N'1982-12-17T00:00:00.000' AS DateTime), 210, 156, NULL, 241, 1, 7, 116, 5, 21000000, 177200000, 10, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (486, N'An Officer and a Gentleman', CAST(N'1982-08-13T00:00:00.000' AS DateTime), 248, 157, NULL, 241, 1, 2, 122, 5, 6000000, 129800000, 6, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (487, N'Rocky III', CAST(N'1982-05-28T00:00:00.000' AS DateTime), 242, 22, NULL, 241, 1, 22, 100, 2, 17000000, 270000000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (488, N'Porky''s', CAST(N'1982-04-16T00:00:00.000' AS DateTime), 249, 158, NULL, 262, 1, 7, 98, 6, 25000000, 105000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (489, N'48 Hrs.', CAST(N'1982-12-08T00:00:00.000' AS DateTime), 250, 2, NULL, 241, 1, 4, 96, 6, 12000000, 78900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (490, N'Poltergeist', CAST(N'1982-06-04T00:00:00.000' AS DateTime), 251, 11, NULL, 241, 1, 10, 114, 5, 10700000, 121700000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (492, N'Annie', CAST(N'1982-06-18T00:00:00.000' AS DateTime), 252, 153, NULL, 241, 1, 20, 128, 1, 50000000, 57000000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (494, N'Superman II', CAST(N'1981-04-09T00:00:00.000' AS DateTime), 313, 83, NULL, 240, 1, 4, 127, 2, 54000000, 190400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (495, N'Arthur', CAST(N'1981-07-17T00:00:00.000' AS DateTime), 253, 151, NULL, 241, 1, 7, 97, 5, 7000000, 95461682, 4, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (497, N'The Cannonball Run', CAST(N'1981-06-19T00:00:00.000' AS DateTime), 254, 103, NULL, 241, 1, 7, 95, 2, 18000000, 72179579, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (498, N'Chariots of Fire', CAST(N'1981-03-30T00:00:00.000' AS DateTime), 255, 97, NULL, 240, 1, 22, 124, 1, 5500000, 59000000, 7, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (500, N'Time Bandits', CAST(N'1981-07-10T00:00:00.000' AS DateTime), 256, 159, NULL, 240, 1, 12, 113, 2, 5000000, 42365581, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (501, N'9 to 5', CAST(N'1980-12-19T00:00:00.000' AS DateTime), 257, 4, NULL, 241, 1, 7, 110, 1, 10000000, 103290500, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (502, N'Stir Crazy', CAST(N'1980-12-12T00:00:00.000' AS DateTime), 258, 5, NULL, 241, 1, 7, 111, 5, 10000000, 101300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (503, N'Airplane!', CAST(N'1980-07-02T00:00:00.000' AS DateTime), 222, 2, NULL, 241, 1, 7, 87, 5, 3500000, 130000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (504, N'Any Which Way You Can', CAST(N'1980-12-17T00:00:00.000' AS DateTime), 259, 31, NULL, 241, 1, 4, 116, 2, 15000000, 70700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (505, N'Private Benjamin', CAST(N'1980-10-10T00:00:00.000' AS DateTime), 260, 6, NULL, 241, 1, 7, 109, 5, 15000000, 69847348, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (507, N'Smokey and the Bandit II', CAST(N'1980-08-15T00:00:00.000' AS DateTime), 254, 153, NULL, 241, 1, 7, 100, 2, 17000000, 66100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (509, N'The Blues Brothers', CAST(N'1980-06-20T00:00:00.000' AS DateTime), 229, 1, NULL, 241, 1, 7, 132, 5, 30000000, 115200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (510, N'Kramer vs. Kramer', CAST(N'1979-12-19T00:00:00.000' AS DateTime), 261, 5, NULL, 241, 1, 2, 105, 2, 8000000, 106260000, 9, 5);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (511, N'The Amityville Horror', CAST(N'1979-07-27T00:00:00.000' AS DateTime), 262, 96, NULL, 241, 1, 10, 118, 5, 4700000, 86400000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (512, N'Rocky II', CAST(N'1979-06-15T00:00:00.000' AS DateTime), 242, 22, NULL, 241, 1, 2, 119, 2, 7000000, 200100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (513, N'Apocalypse Now', CAST(N'1979-08-15T00:00:00.000' AS DateTime), 219, 160, NULL, 241, 1, 5, 153, 6, 31500000, 150000000, 8, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (514, N'10', CAST(N'1979-10-05T00:00:00.000' AS DateTime), 263, 151, NULL, 241, 1, 27, 122, 6, 7000000, 74800000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (515, N'The Jerk', CAST(N'1979-12-14T00:00:00.000' AS DateTime), 264, 1, NULL, 241, 1, 7, 94, 5, 4000000, 73700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (516, N'The Muppet Movie', CAST(N'1979-05-31T00:00:00.000' AS DateTime), 265, 161, NULL, 241, 1, 26, 97, 1, 8000000, 76600000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (517, N'Grease', CAST(N'1978-06-16T00:00:00.000' AS DateTime), 266, 2, NULL, 241, 1, 20, 110, 2, 6000000, 395000000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (518, N'National Lampoon''s Animal House', CAST(N'1978-07-28T00:00:00.000' AS DateTime), 229, 1, NULL, 241, 1, 7, 109, 5, 3000000, 141600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (519, N'Every Which Way but Loose', CAST(N'1978-12-20T00:00:00.000' AS DateTime), 267, 31, NULL, 241, 1, 4, 114, 5, 5000000, 85000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (522, N'Jaws 2', CAST(N'1978-06-16T00:00:00.000' AS DateTime), 268, 1, NULL, 241, 1, 9, 116, 2, 20000000, 208900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (523, N'Dawn of the Dead', CAST(N'1978-09-01T00:00:00.000' AS DateTime), 269, 162, NULL, 241, 1, 10, 127, 6, 1500000, 55000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (524, N'Revenge of the Pink Panther', CAST(N'1978-07-19T00:00:00.000' AS DateTime), 263, 163, NULL, 240, 1, 7, 98, 2, 12000000, 49500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (525, N'The Deer Hunter', CAST(N'1978-12-08T00:00:00.000' AS DateTime), 270, 164, NULL, 241, 1, 2, 183, 6, 15000000, 49000000, 9, 5);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (526, N'Smokey and the Bandit', CAST(N'1977-05-27T00:00:00.000' AS DateTime), 254, 153, NULL, 241, 1, 7, 96, 2, 4300000, 300000000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (527, N'Close Encounters of the Third Kind', CAST(N'1977-11-16T00:00:00.000' AS DateTime), 4, 164, NULL, 241, 1, 3, 135, 2, 20000000, 303800000, 8, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (529, N'Saturday Night Fever', CAST(N'1977-12-14T00:00:00.000' AS DateTime), 236, 2, NULL, 241, 1, 2, 118, 6, 3500000, 237100000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (531, N'A Bridge Too Far', CAST(N'1977-06-15T00:00:00.000' AS DateTime), 101, 165, NULL, 240, 1, 5, 176, 5, 25000000, 50750000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (532, N'The Deep', CAST(N'1977-06-17T00:00:00.000' AS DateTime), 271, 164, NULL, 241, 1, 8, 124, 5, 9000000, 47300000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (533, N'Annie Hall', CAST(N'1977-04-20T00:00:00.000' AS DateTime), 272, 22, NULL, 241, 1, 27, 93, 5, 4000000, 38300000, 5, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (534, N'Rocky', CAST(N'1976-11-21T00:00:00.000' AS DateTime), 238, 22, NULL, 241, 1, 2, 119, 2, 1100000, 225000000, 10, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (537, N'All the President''s Men', CAST(N'1976-04-09T00:00:00.000' AS DateTime), 214, 6, NULL, 241, 1, 9, 138, 5, 8500000, 70600000, 8, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (538, N'The Omen', CAST(N'1976-06-06T00:00:00.000' AS DateTime), 30, 4, NULL, 241, 1, 10, 111, 6, 2800000, 60900000, 2, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (540, N'Silver Streak', CAST(N'1976-12-08T00:00:00.000' AS DateTime), 273, 4, NULL, 241, 1, 7, 114, 2, 6500000, 51100000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (541, N'The Enforcer', CAST(N'1976-12-22T00:00:00.000' AS DateTime), 267, 6, NULL, 241, 1, 4, 96, 6, 9000000, 46236000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (542, N'Midway', CAST(N'1976-06-18T00:00:00.000' AS DateTime), 274, 1, NULL, 241, 1, 5, 131, 2, 3000000, 43200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (543, N'Jaws', CAST(N'1975-06-20T00:00:00.000' AS DateTime), 4, 1, NULL, 241, 1, 9, 124, 2, 9000000, 470700000, 4, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (544, N'The Rocky Horror Picture Show', CAST(N'1975-08-14T00:00:00.000' AS DateTime), 275, 4, NULL, 240, 1, 20, 100, 5, 1400000, 140200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (545, N'One Flew Over the Cuckoo''s Nest', CAST(N'1975-11-19T00:00:00.000' AS DateTime), 276, 22, NULL, 241, 1, 2, 133, 6, 3000000, 109000000, 9, 5);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (546, N'Dog Day Afternoon', CAST(N'1975-09-21T00:00:00.000' AS DateTime), 277, 6, NULL, 241, 1, 14, 125, 5, 1800000, 50000000, 6, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (547, N'Shampoo', CAST(N'1975-02-11T00:00:00.000' AS DateTime), 278, 5, NULL, 241, 1, 2, 110, 6, 4000000, 60000000, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (548, N'The Return of the Pink Panther', CAST(N'1975-05-21T00:00:00.000' AS DateTime), 263, 22, NULL, 240, 1, 7, 114, 1, 5000000, 41800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (553, N'Blazing Saddles', CAST(N'1974-02-07T00:00:00.000' AS DateTime), 279, 6, NULL, 241, 1, 7, 92, 5, 2600000, 119600000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (554, N'The Towering Inferno', CAST(N'1974-12-14T00:00:00.000' AS DateTime), 100, 4, NULL, 241, 1, 25, 165, 5, 14300000, 139700000, 8, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (556, N'Young Frankenstein', CAST(N'1974-12-15T00:00:00.000' AS DateTime), 279, 4, NULL, 241, 1, 7, 105, 5, 2780000, 86200000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (557, N'Earthquake', CAST(N'1974-11-15T00:00:00.000' AS DateTime), 281, 1, NULL, 241, 1, 25, 121, 2, 7000000, 79666653, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (558, N'The Godfather Part II', CAST(N'1974-12-12T00:00:00.000' AS DateTime), 219, 2, NULL, 241, 1, 14, 202, 6, 13000000, 57300000, 11, 6);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (559, N'Airport 1975', CAST(N'1974-10-18T00:00:00.000' AS DateTime), 274, 1, NULL, 241, 1, 25, 106, 2, 3000000, 47285152, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (561, N'The Longest Yard', CAST(N'1974-08-21T00:00:00.000' AS DateTime), 282, 2, NULL, 241, 1, 7, 121, 5, 2900000, 43008075, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (563, N'The Sting', CAST(N'1973-12-26T00:00:00.000' AS DateTime), 283, 1, NULL, 241, 1, 14, 129, 2, 5500000, 159600000, 10, 7);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (564, N'The Exorcist', CAST(N'1973-12-26T00:00:00.000' AS DateTime), 284, 6, NULL, 241, 1, 10, 121, 6, 12000000, 441300000, 10, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (565, N'American Graffiti', CAST(N'1973-08-02T00:00:00.000' AS DateTime), 7, 13, NULL, 241, 1, 2, 112, 2, 777000, 140000000, 5, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (566, N'Papillon', CAST(N'1973-12-16T00:00:00.000' AS DateTime), 285, 5, NULL, 241, 1, 14, 150, 5, 13500000, 53267000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (568, N'Magnum Force', CAST(N'1973-12-25T00:00:00.000' AS DateTime), 443, 31, NULL, 241, 1, 4, 124, 6, NULL, 44680473, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (570, N'Robin Hood', CAST(N'1973-11-08T00:00:00.000' AS DateTime), 287, 3, NULL, 241, 1, 11, 83, 1, 5000000, 32000000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (572, N'The Godfather', CAST(N'1972-03-24T00:00:00.000' AS DateTime), 219, 2, NULL, 241, 1, 14, 177, 6, 7000000, 245100000, 11, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (573, N'The Poseidon Adventure', CAST(N'1972-12-12T00:00:00.000' AS DateTime), 288, 4, NULL, 241, 1, 25, 117, 2, 4700000, 127300000, 8, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (576, N'Deliverance', CAST(N'1972-07-30T00:00:00.000' AS DateTime), 289, 6, NULL, 241, 1, 9, 110, 6, 2000000, 4610000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (578, N'Cabaret', CAST(N'1972-02-13T00:00:00.000' AS DateTime), 290, 166, NULL, 241, 1, 20, 124, 5, 2285000, 42765000, 10, 8);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (582, N'Fiddler on the Roof', CAST(N'1971-11-03T00:00:00.000' AS DateTime), 234, 167, NULL, 241, 1, 20, 179, 1, 9000000, 83300000, 8, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (583, N'The French Connection', CAST(N'1971-10-09T00:00:00.000' AS DateTime), 284, 4, NULL, 241, 1, 14, 104, 6, 1800000, 51700000, 8, 5);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (584, N'Dirty Harry', CAST(N'1971-12-23T00:00:00.000' AS DateTime), 291, 31, NULL, 241, 1, 9, 102, 6, 4000000, 36000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (588, N'A Clockwork Orange', CAST(N'1971-12-19T00:00:00.000' AS DateTime), 292, 168, NULL, 240, 1, 2, 136, 6, 2200000, 26600000, 4, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (589, N'Bedknobs and Broomsticks', CAST(N'1971-10-07T00:00:00.000' AS DateTime), 293, 3, NULL, 240, 1, 8, 117, 1, 20000000, 18530000, 5, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (590, N'Love Story', CAST(N'1970-12-16T00:00:00.000' AS DateTime), 273, 2, NULL, 241, 1, 6, 100, 2, 2200000, 136400000, 7, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (591, N'Airport', CAST(N'1970-03-05T00:00:00.000' AS DateTime), 294, 1, NULL, 241, 1, 25, 137, 2, 10200000, 100500000, 10, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (592, N'MASH', CAST(N'1970-01-25T00:00:00.000' AS DateTime), 295, 4, NULL, 241, 1, 7, 116, 5, NULL, NULL, 5, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (593, N'Patton', CAST(N'1970-04-02T00:00:00.000' AS DateTime), 285, 4, NULL, 241, 1, 19, 170, 2, 12600000, 61800000, 10, 7);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (598, N'Tora! Tora! Tora!', CAST(N'1970-09-23T00:00:00.000' AS DateTime), 297, 4, NULL, 118, 1, 5, 144, 1, 25500000, 29500000, 5, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (600, N'Butch Cassidy and the Sundance Kid', CAST(N'1969-10-24T00:00:00.000' AS DateTime), 283, 4, NULL, 241, 1, 1, 110, 2, 6000000, 102300000, 7, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (601, N'Midnight Cowboy', CAST(N'1969-05-25T00:00:00.000' AS DateTime), 298, 22, NULL, 241, 1, 2, 113, 6, 3200000, 44800000, 7, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (602, N'Easy Rider', CAST(N'1969-07-14T00:00:00.000' AS DateTime), 299, 5, NULL, 241, 1, 2, 95, 6, 360000, 60000000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (605, N'Paint Your Wagon', CAST(N'1969-10-15T00:00:00.000' AS DateTime), 300, 31, NULL, 241, 1, 20, 154, 2, 20000000, 31678778, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (606, N'True Grit', CAST(N'1969-06-11T00:00:00.000' AS DateTime), 301, 2, NULL, 241, 1, 1, 128, 1, NULL, 43700000, 2, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (609, N'2001: A Space Odyssey', CAST(N'1968-04-03T00:00:00.000' AS DateTime), 292, 240, NULL, 240, 1, 3, 142, 1, 12000000, 190000000, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (611, N'The Love Bug', CAST(N'1968-12-24T00:00:00.000' AS DateTime), 293, 3, NULL, 241, 1, 7, 108, 1, 5000000, 51264000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (612, N'The Odd Couple', CAST(N'1968-05-02T00:00:00.000' AS DateTime), 302, 2, NULL, 241, 1, 7, 105, 2, 1200000, 44527234, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (613, N'Bullitt', CAST(N'1968-10-17T00:00:00.000' AS DateTime), 271, 6, N'Best. Car chase. Ever.', 241, 1, 14, 113, 5, 5500000, 42300000, 2, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (614, N'Romeo and Juliet', CAST(N'1968-10-08T00:00:00.000' AS DateTime), 303, 96, NULL, 240, 1, 6, 138, 2, 850000, 38900000, 4, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (615, N'Oliver!', CAST(N'1968-09-26T00:00:00.000' AS DateTime), 304, 5, N'Not seen it.', 240, 1, 20, 153, 1, 10000000, 77400000, 11, 5);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (617, N'Planet of the Apes', CAST(N'1968-04-03T00:00:00.000' AS DateTime), 285, 4, NULL, 241, 1, 3, 112, 2, 5800000, 33400000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (618, N'The Graduate', CAST(N'1967-12-21T00:00:00.000' AS DateTime), 306, 22, N'Not seen it.', 241, 1, 7, 105, 5, 3000000, 104900000, 7, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (619, N'The Jungle Book', CAST(N'1967-10-18T00:00:00.000' AS DateTime), 287, 3, NULL, 241, 1, 11, 78, 1, 4000000, 205800000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (621, N'Bonnie and Clyde', CAST(N'1967-08-13T00:00:00.000' AS DateTime), 307, 6, N'Not seen it.', 241, 1, 14, 111, 6, 2500000, 70000000, 10, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (622, N'The Dirty Dozen', CAST(N'1967-06-15T00:00:00.000' AS DateTime), 282, 11, NULL, 241, 1, 5, 150, 5, 5400000, 45300000, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (627, N'Night of the Living Dead', CAST(N'1968-10-01T00:00:00.000' AS DateTime), 269, 162, N'Not seen it.', 241, 1, 10, 96, 5, 114000, 30000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (633, N'The Good, the Bad and the Ugly', CAST(N'1966-12-15T00:00:00.000' AS DateTime), 308, 34, NULL, 264, 12, 1, 177, 6, 1200000, 25100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (636, N'Grand Prix', CAST(N'1966-12-21T00:00:00.000' AS DateTime), 309, 11, NULL, 241, 1, 22, 179, 2, 9000000, 20800000, 3, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (638, N'The Sound of Music', CAST(N'1965-03-02T00:00:00.000' AS DateTime), 89, 4, N'Not seen it.', 241, 1, 20, 174, 1, 8200000, 286200000, 10, 5);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (639, N'Doctor Zhivago', CAST(N'1965-12-22T00:00:00.000' AS DateTime), 310, 11, N'Not seen it.', 240, 1, 6, 193, 5, 11000000, 111700000, 10, 5);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (644, N'What''s New Pussycat?', CAST(N'1965-06-22T00:00:00.000' AS DateTime), 311, 22, N'Not seen it.', 241, 1, 7, 108, 5, NULL, 18820000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (646, N'Von Ryan''s Express', CAST(N'1965-06-23T00:00:00.000' AS DateTime), 281, 4, N'Not seen it.', 241, 1, 5, 117, 2, 5760000, 17111111, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (647, N'Mary Poppins', CAST(N'1964-08-27T00:00:00.000' AS DateTime), 293, 3, N'Worst. Cockney accent. Ever.', 241, 1, 20, 139, 1, NULL, 102300000, 13, 5);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (648, N'My Fair Lady', CAST(N'1964-11-09T00:00:00.000' AS DateTime), 312, 6, N'Not seen it.', 241, 1, 20, 170, 1, 17000000, 72000000, 12, 8);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (650, N'A Fistful of Dollars', CAST(N'1964-09-12T00:00:00.000' AS DateTime), 308, 34, NULL, 264, 12, 1, 99, 5, 225000, 14500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (652, N'A Shot in the Dark', CAST(N'1964-06-23T00:00:00.000' AS DateTime), 263, 167, N'Not seen it.', 240, 1, 7, 102, 2, NULL, 12368234, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (653, N'A Hard Day''s Night', CAST(N'1964-07-06T00:00:00.000' AS DateTime), 313, 22, N'Not seen it.', 240, 1, 20, 87, 1, 200000, 12299668, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (655, N'Cleopatra', CAST(N'1963-06-12T00:00:00.000' AS DateTime), 314, 4, N'Not seen it.', 241, 1, 13, 248, 2, 31100000, 57800000, 9, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (656, N'How the West Was Won', CAST(N'1963-02-20T00:00:00.000' AS DateTime), 315, 11, N'Not seen it.', 241, 1, 1, 164, 1, 14483000, 50000000, 8, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (657, N'It''s a Mad, Mad, Mad, Mad World', CAST(N'1963-11-07T00:00:00.000' AS DateTime), 316, 22, N'Not seen it.', 241, 1, 7, 210, 1, 9400000, 60000000, 6, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (660, N'The Sword in the Stone', CAST(N'1963-12-25T00:00:00.000' AS DateTime), 287, 3, N'Not seen it.', 241, 1, 11, 79, 1, 3000000, 22200000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (662, N'The Birds', CAST(N'1963-03-28T00:00:00.000' AS DateTime), 317, 169, N'Not seen it.', 241, 1, 10, 119, 5, 3300000, 11400000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (664, N'Lawrence of Arabia', CAST(N'1962-12-10T00:00:00.000' AS DateTime), 310, 170, N'Not seen it.', 240, 1, 19, 222, 2, 15000000, 70000000, 10, 7);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (665, N'The Longest Day', CAST(N'1962-09-25T00:00:00.000' AS DateTime), 318, 4, N'Not seen it.', 241, 1, 5, 178, 2, 7750000, 50100000, 5, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (669, N'Mutiny on the Bounty', CAST(N'1962-11-08T00:00:00.000' AS DateTime), 319, 11, N'Not seen it.', 241, 1, 13, 185, 5, 19000000, 13680000, 7, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (670, N'To Kill a Mockingbird', CAST(N'1962-12-25T00:00:00.000' AS DateTime), 320, 1, NULL, 241, 1, 2, 129, 2, 2000000, 13100000, 8, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (674, N'West Side Story', CAST(N'1961-10-18T00:00:00.000' AS DateTime), 89, 167, N'Not seen it.', 241, 1, 20, 152, 2, 6000000, 43700000, 11, 10);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (675, N'The Guns of Navarone', CAST(N'1961-04-27T00:00:00.000' AS DateTime), 321, 5, NULL, 240, 1, 5, 158, 2, 6000000, 28900000, 7, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (676, N'El Cid', CAST(N'1961-12-06T00:00:00.000' AS DateTime), 322, 171, N'Not seen it.', 241, 1, 13, 184, 1, 6200000, 30000000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (677, N'The Absent-Minded Professor', CAST(N'1961-03-16T00:00:00.000' AS DateTime), 293, 3, N'Not seen it.', 241, 1, 7, 97, 1, NULL, 25400000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (678, N'The Parent Trap', CAST(N'1961-06-21T00:00:00.000' AS DateTime), 323, 3, N'Not seen it.', 241, 1, 7, 128, 1, NULL, 25100000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (679, N'La Dolce Vita', CAST(N'1960-02-05T00:00:00.000' AS DateTime), 324, 172, N'Not seen it.', 264, 12, 7, 174, 5, NULL, 19500000, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (682, N'One Hundred and One Dalmatians', CAST(N'1961-01-25T00:00:00.000' AS DateTime), 287, 3, N'Not seen it.', 241, 1, 11, 79, 1, 4000000, 215800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (684, N'Spartacus', CAST(N'1960-10-07T00:00:00.000' AS DateTime), 292, 173, NULL, 241, 1, 13, 184, 2, 12000000, 60000000, 6, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (685, N'Psycho', CAST(N'1960-06-16T00:00:00.000' AS DateTime), 317, 174, N'Not seen it.', 241, 1, 10, 109, 5, 806947, 50000000, 4, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (687, N'Swiss Family Robinson', CAST(N'1960-12-21T00:00:00.000' AS DateTime), 318, 3, N'Not seen it.', 241, 1, 8, 126, 1, 4000000, 40000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (688, N'The Alamo', CAST(N'1960-10-24T00:00:00.000' AS DateTime), 325, 22, N'Not seen it.', 241, 1, 5, 167, 1, 12000000, 17260000, 7, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (691, N'Ocean''s 11', CAST(N'1960-08-10T00:00:00.000' AS DateTime), 319, 6, N'Not seen it.', 241, 1, 7, 127, 2, 2800000, 5650000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (694, N'Ben-Hur', CAST(N'1959-11-18T00:00:00.000' AS DateTime), 326, 11, NULL, 241, 1, 2, 212, 2, 15200000, 146900000, 12, 11);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (695, N'Sleeping Beauty', CAST(N'1959-01-29T00:00:00.000' AS DateTime), 287, 3, N'Not seen it.', 241, 1, 11, 75, 1, 6000000, 51600000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (696, N'North by Northwest', CAST(N'1959-07-28T00:00:00.000' AS DateTime), 317, 11, N'Not seen it.', 241, 1, 9, 136, 2, 4326000, 9800000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (697, N'Some Like It Hot', CAST(N'1959-03-29T00:00:00.000' AS DateTime), 327, 167, N'Not seen it.', 241, 1, 27, 121, 1, 2900000, 40000000, 6, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (702, N'Rio Bravo', CAST(N'1959-03-18T00:00:00.000' AS DateTime), 328, 6, N'Not seen it.', 241, 1, 1, 141, 2, NULL, 12535000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (704, N'South Pacific', CAST(N'1958-03-19T00:00:00.000' AS DateTime), 300, 4, N'Not seen it.', 241, 1, 20, 171, 1, 5610000, 36800000, 3, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (706, N'Cat on a Hot Tin Roof', CAST(N'1958-09-20T00:00:00.000' AS DateTime), 329, 11, N'Not seen it.', 241, 1, 2, 107, 5, NULL, NULL, 6, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (709, N'The Vikings', CAST(N'1958-06-28T00:00:00.000' AS DateTime), 297, 22, N'Not seen it.', 241, 1, 8, 116, 2, 3500000, 6000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (710, N'Vertigo', CAST(N'1958-05-09T00:00:00.000' AS DateTime), 317, 2, N'Not seen it.', 241, 1, 9, 128, 2, 2500000, 7300000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (714, N'The Bridge on the River Kwai', CAST(N'1957-10-02T00:00:00.000' AS DateTime), 310, 170, NULL, 240, 1, 5, 161, 1, 2800000, 30600000, 8, 7);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (717, N'Old Yeller', CAST(N'1957-12-25T00:00:00.000' AS DateTime), 293, 3, N'Not seen it.', 241, 1, 2, 83, 1, NULL, 6250000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (721, N'A Farewell to Arms', CAST(N'1957-12-14T00:00:00.000' AS DateTime), 330, 175, N'Not seen it.', 241, 1, 2, 152, 5, 4300000, 20000000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (722, N'Gunfight at the O.K. Corral', CAST(N'1957-05-30T00:00:00.000' AS DateTime), 331, 2, N'Not seen it.', 241, 1, 1, 122, 2, 2000000, 10700000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (724, N'The Ten Commandments', CAST(N'1956-10-05T00:00:00.000' AS DateTime), 127, 144, N'Not seen it.', 241, 1, 12, 220, 1, 13200000, 80000000, 7, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (726, N'War and Peace', CAST(N'1956-08-21T00:00:00.000' AS DateTime), 332, 96, N'Not seen it.', 241, 1, 2, 208, 2, 6000000, 12500000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (727, N'The King and I', CAST(N'1956-06-28T00:00:00.000' AS DateTime), 333, 4, N'Not seen it.', 241, 1, 20, 133, 1, 4550000, 21300000, 9, 5);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (728, N'The Searchers', CAST(N'1956-03-13T00:00:00.000' AS DateTime), 315, 6, N'Not seen it', 241, 1, 1, 119, 1, 3750000, 37500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (731, N'High Society', CAST(N'1956-07-17T00:00:00.000' AS DateTime), 334, 176, N'Not seen it.', 241, 1, 20, 111, 1, 2762000, 8258000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (736, N'Oklahoma!', CAST(N'1955-10-11T00:00:00.000' AS DateTime), 335, 37, N'Not seen it.', 241, 1, 20, 145, 1, 5000000, 7100000, 4, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (737, N'Guys and Dolls', CAST(N'1955-11-03T00:00:00.000' AS DateTime), 314, 11, N'Not seen it.', 241, 1, 20, 150, 1, 5500000, 20000000, 4, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (738, N'Lady and the Tramp', CAST(N'1955-06-22T00:00:00.000' AS DateTime), 336, 3, NULL, 241, 1, 11, 75, 1, 4000000, 93600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (744, N'20,000 Leagues Under the Sea', CAST(N'1954-12-23T00:00:00.000' AS DateTime), 297, 3, N'Not seen it.', 241, 1, 8, 127, 1, 5000000, 28200000, 3, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (753, N'The Robe', CAST(N'1953-09-16T00:00:00.000' AS DateTime), 337, 4, N'Not seen it.', 241, 1, 2, 135, 1, 4100000, 36000000, 5, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (754, N'From Here to Eternity', CAST(N'1953-08-05T00:00:00.000' AS DateTime), 335, 5, N'Not seen it.', 241, 1, 2, 118, 2, 2500000, 30500000, 13, 8);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (755, N'Shane', CAST(N'1953-04-23T00:00:00.000' AS DateTime), 338, 2, N'Not seen it.', 241, 1, 1, 118, 2, 3100000, 20000000, 6, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (757, N'Peter Pan', CAST(N'1953-02-05T00:00:00.000' AS DateTime), 336, 3, N'Not seen it.', 241, 1, 11, 76, 1, 4000000, 87400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (761, N'Gentlemen Prefer Blondes', CAST(N'1953-07-15T00:00:00.000' AS DateTime), 328, 4, N'Not seen it.', 241, 1, 20, 91, 1, 2260000, 5300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (766, N'Ivanhoe', CAST(N'1952-07-31T00:00:00.000' AS DateTime), 339, 11, N'Not seen it.', 240, 1, 8, 106, 1, 3842000, 10878000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (767, N'Singin'' in the Rain', CAST(N'1952-04-11T00:00:00.000' AS DateTime), 340, 11, N'Some guy sings. In the rain.', 241, 1, 20, 103, 1, 2500000, 12400000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (769, N'Moulin Rouge', CAST(N'1952-12-23T00:00:00.000' AS DateTime), 252, 177, NULL, 240, 1, 2, 119, 2, 1500000, 5000000, 7, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (774, N'Alice in Wonderland', CAST(N'1951-06-26T00:00:00.000' AS DateTime), 336, 3, N'Not seen it.', 241, 1, 11, 75, 1, 3000000, 5232000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (776, N'A Streetcar Named Desire', CAST(N'1951-09-18T00:00:00.000' AS DateTime), 341, 6, N'Not seen it.', 241, 1, 2, 122, 5, 1800000, 8000000, 12, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (779, N'The African Queen', CAST(N'1951-12-23T00:00:00.000' AS DateTime), 252, 170, N'Not seen it.', 241, 1, 8, 105, 1, 1000000, 10750000, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (781, N'Strangers on a Train', CAST(N'1951-06-30T00:00:00.000' AS DateTime), 317, 178, N'Not seen it.', 241, 1, 9, 101, 2, 1200000, 7000000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (783, N'Cinderella', CAST(N'1950-03-04T00:00:00.000' AS DateTime), 336, 3, N'Not seen it.', 241, 1, 11, 75, 1, 2900000, 263600000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (784, N'King Solomon''s Mines', CAST(N'1950-11-24T00:00:00.000' AS DateTime), 342, 11, N'Not seen it.', 241, 1, 8, 103, 2, 2300000, 15100000, 3, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (785, N'Annie Get Your Gun', CAST(N'1950-05-17T00:00:00.000' AS DateTime), 334, 11, N'Not seen it.', 241, 1, 20, 107, 1, 3734000, 7756000, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (796, N'Sands of Iwo Jima', CAST(N'1949-12-14T00:00:00.000' AS DateTime), 138, 95, N'Not seen it.', 241, 1, 5, 100, 2, 1000000, 5000000, 4, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (853, N'For Whom the Bell Tolls', CAST(N'1943-07-14T00:00:00.000' AS DateTime), 344, 2, N'Not seen it.', 241, 1, 5, 170, 1, 3000000, 7100000, 9, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (858, N'Casablanca', CAST(N'1942-11-26T00:00:00.000' AS DateTime), 345, 6, N'Not seen it.', 241, 1, 2, 102, 1, 878000, 3700000, 8, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (885, N'The Great Dictator', CAST(N'1940-10-15T00:00:00.000' AS DateTime), 348, 180, N'Not seen it.', 241, 1, 7, 124, 1, 2000000, 5000000, 5, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (887, N'The Grapes of Wrath', CAST(N'1940-01-24T00:00:00.000' AS DateTime), 315, 4, N'Depressing.', 241, 1, 2, 129, 2, 800000, 2500000, 7, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (893, N'Gone with the Wind', CAST(N'1939-12-15T00:00:00.000' AS DateTime), 349, 175, N'Not seen it.', 241, 1, 6, 221, 2, 3850000, 198000000, 13, 8);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (899, N'Goodbye, Mr. Chips', CAST(N'1939-05-15T00:00:00.000' AS DateTime), 126, 11, N'Not seen it.', 240, 1, 2, 114, 1, 1051000, 3252000, 7, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (900, N'The Wizard of Oz', CAST(N'1939-08-25T00:00:00.000' AS DateTime), 349, 11, NULL, 241, 1, 12, 101, 1, 2800000, 22300000, 6, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (911, N'Angels with Dirty Faces', CAST(N'1938-11-26T00:00:00.000' AS DateTime), 345, 6, N'Not seen it.', 241, 1, 14, 97, 2, NULL, 1700000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (913, N'Snow White and the Seven Dwarfs', CAST(N'1937-12-21T00:00:00.000' AS DateTime), 351, 3, NULL, 241, 1, 11, 83, 1, 2000000, 184925485, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (941, N'Anna Karenina', CAST(N'1935-08-30T00:00:00.000' AS DateTime), 354, 11, N'Not seen it.', 241, 1, 2, 95, 1, 1152000, 2304000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (972, N'Frankenstein', CAST(N'1931-11-21T00:00:00.000' AS DateTime), 356, 1, N'Not seen it.', 241, 1, 10, 71, 2, 262000, 12000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (977, N'Dracula', CAST(N'1931-02-12T00:00:00.000' AS DateTime), 357, 1, N'Not seen it.', 241, 1, 10, 85, 2, 355000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (982, N'All Quiet on the Western Front', CAST(N'1930-04-21T00:00:00.000' AS DateTime), 319, 1, N'Not seen it.', 241, 1, 5, 152, 2, 1200000, 3000000, 4, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1063, N'The Kid', CAST(N'1921-01-21T00:00:00.000' AS DateTime), 348, 180, N'Not seen it.', 241, 10, 7, 68, 1, 250000, 2500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1120, N'Star Wars: The Force Awakens', CAST(N'2015-12-17T00:00:00.000' AS DateTime), 44, 13, NULL, 241, 1, 3, 136, 4, 306000000, 2066000000, 5, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1121, N'Jurassic World', CAST(N'2015-06-12T00:00:00.000' AS DateTime), 148, 41, NULL, 241, 1, 8, 124, 4, 150000000, 1670000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1122, N'Furious 7', CAST(N'2015-04-03T00:00:00.000' AS DateTime), 147, 54, NULL, 241, 1, 4, 137, 4, 190000000, 1516000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1123, N'Avengers: Age of Ultron', CAST(N'2015-05-01T00:00:00.000' AS DateTime), 86, 53, NULL, 241, 1, 4, 141, 4, 250000000, 1405000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1124, N'Minions', CAST(N'2015-06-11T00:00:00.000' AS DateTime), 141, 124, NULL, 241, 1, 11, 91, 1, 74000000, 1159000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1125, N'Spectre', CAST(N'2015-10-26T00:00:00.000' AS DateTime), 61, 51, N'Daniel Craig discovers that all of his bad luck has been caused by a guy with severe daddy issues. Seriously? Evil megalomaniacs used to be cool.', 240, 1, 9, 148, 4, 245000000, 880700000, 1, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1126, N'Inside Out', CAST(N'2015-06-19T00:00:00.000' AS DateTime), 46, 10, NULL, 241, 1, 11, 94, 1, 175000000, 857400000, 2, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1127, N'Mission: Impossible - Rogue Nation', CAST(N'2015-07-31T00:00:00.000' AS DateTime), 146, 59, NULL, 241, 1, 4, 131, 4, 150000000, 682300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1128, N'The Hunger Games: Mockingjay - Part 2', CAST(N'2015-11-20T00:00:00.000' AS DateTime), 23, 120, NULL, 241, 1, 3, 137, 4, 150000000, 653400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1129, N'The Martian', CAST(N'2015-10-02T00:00:00.000' AS DateTime), 51, 64, N'Matt Damon is basically MacGuyver in space. I like it when he sciences the sh*t out of stuff.', 241, 1, 3, 141, 4, 108000000, 630200000, 7, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1130, N'Captain America: Civil War', CAST(N'2016-05-06T00:00:00.000' AS DateTime), 132, 53, NULL, 241, 1, 4, 147, 4, 250000000, 1153000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1131, N'Finding Dory', CAST(N'2016-06-17T00:00:00.000' AS DateTime), 170, 10, NULL, 241, 1, 11, 97, 1, 200000000, 1025000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1132, N'Zootopia', CAST(N'2016-04-04T00:00:00.000' AS DateTime), 178, 3, NULL, 241, 1, 11, 108, 2, 150000000, 1024000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1133, N'The Jungle Book', CAST(N'2016-04-15T00:00:00.000' AS DateTime), 160, 3, NULL, 241, 1, 8, 105, 2, 175000000, 966500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1134, N'Batman v Superman: Dawn of Justice', CAST(N'2016-03-25T00:00:00.000' AS DateTime), 76, 133, NULL, 241, 1, 4, 151, 3, 250000000, 873300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1135, N'The Secret Life of Pets', CAST(N'2016-07-08T00:00:00.000' AS DateTime), 179, 124, NULL, 241, 1, 11, 87, 1, 75000000, 873400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1136, N'Deadpool', CAST(N'2016-02-12T00:00:00.000' AS DateTime), 180, 53, NULL, 241, 1, 4, 108, 5, 58000000, 782600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1137, N'Suicide Squad', CAST(N'2016-08-05T00:00:00.000' AS DateTime), 181, 133, NULL, 241, 1, 4, 123, 5, 175000000, 745600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1138, N'The Mermaid', CAST(N'2016-02-08T00:00:00.000' AS DateTime), 182, 134, NULL, 48, 5, 12, 94, 5, 60720000, 553800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1139, N'X-Men: Apocalypse', CAST(N'2016-05-09T00:00:00.000' AS DateTime), 14, 53, NULL, 241, 1, 4, 144, 4, 178000000, 543900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1140, N'Doctor Strange', CAST(N'2016-11-04T00:00:00.000' AS DateTime), 183, 53, NULL, 241, 1, 4, 115, 4, 165000000, 571500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1141, N'City of God', CAST(N'2002-08-30T00:00:00.000' AS DateTime), 185, 136, NULL, 263, 11, 14, 130, 6, 3300000, 30600000, 4, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1142, N'The Parent Trap', CAST(N'1998-07-29T00:00:00.000' AS DateTime), 188, 3, N'I''ve never seen it but Andy Brown thinks it''s the best movie ever made. Don''t judge him too harshly; he has three children and doesn''t get out much.', 241, 1, 7, 128, 2, 75000000, 83900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1143, N'Captain Blood', CAST(N'1935-12-28T00:00:00.000' AS DateTime), 345, 179, N'Not seen it.', 241, 1, 8, 119, 2, 1200000, 3100000, 5, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1144, N'The Adventures of Robin Hood', CAST(N'1938-05-14T00:00:00.000' AS DateTime), 345, 6, N'Not seen it.', 241, 1, 8, 102, 1, 2033000, 3981000, 4, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1145, N'The Charge of the Light Brigade', CAST(N'1936-10-20T00:00:00.000' AS DateTime), 345, 6, N'Not seen it.', 241, 1, 8, 115, 2, 1330000, 3300000, 3, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1146, N'The Hound of the Baskervilles', CAST(N'1939-03-24T00:00:00.000' AS DateTime), 346, 4, N'Not seen it.', 241, 1, 18, 81, 2, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1147, N'The Adventures of Sherlock Holmes', CAST(N'1939-09-01T00:00:00.000' AS DateTime), 347, 4, N'Not seen it.', 241, 1, 18, 81, 2, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1148, N'Mutiny on the Bounty', CAST(N'1935-11-08T00:00:00.000' AS DateTime), 350, 11, N'Not seen it.', 241, 1, 2, 132, 2, 1950000, 4460000, 8, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1149, N'The Misfits', CAST(N'1961-02-01T00:00:00.000' AS DateTime), 252, 181, N'Not seen it.', 241, 1, 2, 125, 2, 4000000, 4100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1150, N'Lolita', CAST(N'1962-06-13T00:00:00.000' AS DateTime), 292, 181, N'Not seen it.', 240, 1, 2, 152, 5, 2000000, 9250000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1151, N'The 39 Steps', CAST(N'1935-06-06T00:00:00.000' AS DateTime), 317, 18, N'Not seen it.', 240, 1, 9, 86, 1, 100000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1152, N'Citizen Kane', CAST(N'1941-05-01T00:00:00.000' AS DateTime), 352, 182, N'Not seen it.', 241, 1, 2, 119, 1, 839727, 1140769, 9, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1153, N'Metropolis', CAST(N'1927-01-10T00:00:00.000' AS DateTime), 353, 183, N'Not seen it.', 257, 10, 3, 153, 2, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1154, N'Anna Karenina', CAST(N'1948-01-22T00:00:00.000' AS DateTime), 355, 184, N'Not seen it.', 240, 1, 2, 139, 2, 1000000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1155, N'Scarface', CAST(N'1932-04-09T00:00:00.000' AS DateTime), 328, 22, N'Not seen it.', 241, 1, 14, 95, 5, NULL, 600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1156, N'The Public Enemy', CAST(N'1931-04-23T00:00:00.000' AS DateTime), 358, 6, N'Not seen it.', 241, 1, 14, 83, 2, 151000, NULL, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1157, N'It''s a Wonderful Life', CAST(N'1946-12-20T00:00:00.000' AS DateTime), 359, 185, N'Not seen it.', 241, 1, 12, 130, 1, 3180000, 3300000, 5, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1158, N'Mr. Smith Goes to Washington', CAST(N'1939-10-17T00:00:00.000' AS DateTime), 359, 5, N'Not seen it.', 241, 1, 7, 129, 1, 1500000, 9000000, 11, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1159, N'Brief Encounter', CAST(N'1945-11-26T00:00:00.000' AS DateTime), 310, 186, N'Not seen it.', 240, 1, 6, 86, 2, 1000000, NULL, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1160, N'The Killers', CAST(N'1946-08-28T00:00:00.000' AS DateTime), 360, 1, N'Not seen it.', 241, 1, 14, 103, 2, NULL, 2500000, 4, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1161, N'The Seventh Veil', CAST(N'1945-10-18T00:00:00.000' AS DateTime), 342, 187, N'Not seen it.', 240, 1, 14, 94, 2, 100000, 2000000, 1, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1162, N'Great Expectations', CAST(N'1946-12-26T00:00:00.000' AS DateTime), 310, 186, N'Not seen it.', 240, 1, 2, 113, 2, 500000, 2000000, 5, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1163, N'Miracle on 34th Street', CAST(N'1947-05-02T00:00:00.000' AS DateTime), 294, 4, N'Not seen it.', 241, 1, 7, 96, 1, 630000, 2650000, 4, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1164, N'Hamlet', CAST(N'1948-05-04T00:00:00.000' AS DateTime), 361, 188, N'Not seen it.', 240, 1, 2, 155, 1, 750000, 3250000, 7, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1165, N'Joan of Arc', CAST(N'1948-11-11T00:00:00.000' AS DateTime), 349, 189, N'Not seen it.', 241, 1, 19, 145, 1, 4650000, 5768142, 7, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1166, N'The Three Musketeers', CAST(N'1948-10-20T00:00:00.000' AS DateTime), 362, 11, N'Not seen it.', 241, 1, 8, 125, 1, 4474000, 8412000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1167, N'The Big Boss', CAST(N'1971-10-03T00:00:00.000' AS DateTime), 363, 103, NULL, 265, 2, 17, 110, 6, 100000, 2800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1168, N'Fist of Fury', CAST(N'1972-03-22T00:00:00.000' AS DateTime), 363, 103, NULL, 265, 5, 17, 108, 6, 100000, 3400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1169, N'Way of the Dragon', CAST(N'1972-12-30T00:00:00.000' AS DateTime), 364, 103, NULL, 265, 5, 17, 100, 6, 130000, 5200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1170, N'Game of Death', CAST(N'1978-03-23T00:00:00.000' AS DateTime), 365, 103, NULL, 265, 2, 17, 85, 6, 850000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1171, N'Enter the Dragon', CAST(N'1973-07-26T00:00:00.000' AS DateTime), 365, 6, NULL, 265, 1, 17, 102, 6, 850000, 22000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1172, N'Monty Python and the Holy Grail', CAST(N'1975-04-09T00:00:00.000' AS DateTime), 256, 190, NULL, 240, 1, 7, 92, 5, 400000, 5000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1173, N'Monty Python''s Life of Brian', CAST(N'1979-08-17T00:00:00.000' AS DateTime), 366, 190, NULL, 240, 1, 7, 93, 5, 4000000, 20000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1174, N'Drunken Master', CAST(N'1978-10-05T00:00:00.000' AS DateTime), 367, 191, NULL, 265, 2, 17, 110, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1175, N'Half a Loaf of Kung Fu', CAST(N'1978-07-01T00:00:00.000' AS DateTime), 368, 192, NULL, 265, 5, 17, 98, 6, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1176, N'Snake and Crane Arts of Shaolin', CAST(N'1978-03-08T00:00:00.000' AS DateTime), 368, 192, NULL, 265, 5, 17, 93, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1177, N'The Fearless Hyena', CAST(N'1979-02-17T00:00:00.000' AS DateTime), 369, 193, NULL, 265, 2, 17, 97, 6, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1178, N'Snake in the Eagle''s Shadow', CAST(N'1978-03-01T00:00:00.000' AS DateTime), 367, 191, NULL, 265, 2, 17, 98, 6, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1179, N'Spiritual Kung Fu', CAST(N'1978-11-23T00:00:00.000' AS DateTime), 363, 192, NULL, 265, 5, 17, 97, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1180, N'Dragon Fist', CAST(N'1979-04-21T00:00:00.000' AS DateTime), 363, 192, NULL, 265, 5, 17, 93, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1181, N'Magnificent Bodyguards', CAST(N'1978-04-27T00:00:00.000' AS DateTime), 363, 192, NULL, 265, 5, 17, 78, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1182, N'The Young Master', CAST(N'1980-02-09T00:00:00.000' AS DateTime), 369, 103, NULL, 265, 2, 17, 106, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1183, N'Project A', CAST(N'1983-12-22T00:00:00.000' AS DateTime), 369, 103, NULL, 265, 2, 17, 106, 6, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1184, N'Wheels on Meals', CAST(N'1984-08-17T00:00:00.000' AS DateTime), 118, 103, N'Jackie Chan vs Benny "the Jet" Urquidez - best fight scene ever.', 265, 2, 17, 104, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1185, N'Cannonball Run II', CAST(N'1984-06-29T00:00:00.000' AS DateTime), 254, 103, NULL, 241, 1, 7, 108, 2, 20000000, 28000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1186, N'Police Story', CAST(N'1985-12-14T00:00:00.000' AS DateTime), 369, 103, NULL, 265, 2, 17, 101, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1187, N'Winners and Sinners', CAST(N'1983-07-07T00:00:00.000' AS DateTime), 118, 103, NULL, 265, 2, 17, 101, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1188, N'My Lucky Stars', CAST(N'1985-02-10T00:00:00.000' AS DateTime), 118, 103, NULL, 265, 2, 17, 96, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1189, N'Twinkle, Twinkle Lucky Stars', CAST(N'1985-08-15T00:00:00.000' AS DateTime), 118, 103, NULL, 265, 2, 17, 90, 6, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1190, N'Armour of God', CAST(N'1987-01-21T00:00:00.000' AS DateTime), 369, 103, NULL, 265, 2, 17, 97, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1191, N'Project A Part II', CAST(N'1987-08-19T00:00:00.000' AS DateTime), 369, 103, NULL, 265, 2, 17, 106, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1192, N'Police Story 2', CAST(N'1988-08-20T00:00:00.000' AS DateTime), 369, 103, NULL, 265, 2, 17, 101, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1193, N'Dragons Forever', CAST(N'1988-02-11T00:00:00.000' AS DateTime), 118, 103, N'Jackie Chan vs. Benny Urquidez round 2.', 265, 2, 17, 94, 6, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1194, N'Armour of God II: Operation Condor', CAST(N'1991-02-07T00:00:00.000' AS DateTime), 369, 103, NULL, 265, 2, 17, 106, 5, NULL, 10400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1195, N'Police Story 3: Super Cop', CAST(N'1992-07-04T00:00:00.000' AS DateTime), 114, 103, NULL, 265, 2, 17, 95, 5, 900000, 16250000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1196, N'Once Upon a Time in China', CAST(N'1991-08-15T00:00:00.000' AS DateTime), 370, 103, NULL, 265, 2, 17, 134, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1197, N'Twin Dragons', CAST(N'1992-01-15T00:00:00.000' AS DateTime), 370, 103, NULL, 265, 2, 17, 104, 3, 10000000, 11000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1198, N'Once Upon a Time in China II', CAST(N'1992-04-16T00:00:00.000' AS DateTime), 370, 103, NULL, 265, 2, 17, 113, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1199, N'Once Upon a Time in China III', CAST(N'1993-02-11T00:00:00.000' AS DateTime), 370, 103, NULL, 265, 2, 17, 105, 2, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1200, N'Fong Sai-yuk', CAST(N'1993-03-04T00:00:00.000' AS DateTime), 371, 194, NULL, 265, 2, 17, 106, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1201, N'Fong Sai-yuk II', CAST(N'1993-07-30T00:00:00.000' AS DateTime), 371, 194, NULL, 265, 2, 17, 96, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1202, N'Fist of Legend', CAST(N'1994-12-22T00:00:00.000' AS DateTime), 372, 194, N'Remake of Fist of Fury.', 265, 2, 17, 103, 6, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1203, N'Romeo Must Die', CAST(N'2000-03-22T00:00:00.000' AS DateTime), 373, 52, NULL, 241, 1, 4, 115, 5, 25000000, 91000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1204, N'The One', CAST(N'2001-11-02T00:00:00.000' AS DateTime), 374, 19, NULL, 241, 1, 3, 87, 5, 49000000, 72700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1205, N'Kiss of the Dragon', CAST(N'2001-07-06T00:00:00.000' AS DateTime), 375, 195, NULL, 79, 1, 4, 98, 6, 25000000, 64400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1206, N'Cradle 2 the Grave', CAST(N'2003-02-28T00:00:00.000' AS DateTime), 373, 52, NULL, 241, 1, 4, 101, 5, 25000000, 56500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1207, N'Unleashed', CAST(N'2005-02-02T00:00:00.000' AS DateTime), 376, 196, NULL, 79, 1, 17, 102, 6, 45000000, 50900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1208, N'Fearless', CAST(N'2006-01-26T00:00:00.000' AS DateTime), 377, 134, NULL, 48, 5, 17, 105, 5, NULL, 68100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1209, N'The Warlords', CAST(N'2007-12-12T00:00:00.000' AS DateTime), 378, 134, NULL, 265, 5, 5, 127, 5, 40000000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1210, N'Dragon', CAST(N'2011-05-13T00:00:00.000' AS DateTime), 378, 197, NULL, 265, 5, 17, 116, 5, 20000000, 29100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1211, N'The Forbidden Kingdom', CAST(N'2008-04-18T00:00:00.000' AS DateTime), 379, 128, NULL, 241, 1, 17, 104, 4, 55000000, 128000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1212, N'The Mummy: Tomb of the Dragon Emperor', CAST(N'2008-08-01T00:00:00.000' AS DateTime), 32, 128, NULL, 241, 1, 8, 111, 4, 145000000, 401100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1213, N'The Expendables', CAST(N'2010-08-03T00:00:00.000' AS DateTime), 242, 80, NULL, 241, 1, 4, 103, 6, 82000000, 274500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1214, N'The Expendables 2', CAST(N'2012-08-13T00:00:00.000' AS DateTime), 380, 80, NULL, 241, 1, 4, 103, 5, 100000000, 305400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1215, N'The Expendables 3', CAST(N'2014-08-04T00:00:00.000' AS DateTime), 381, 80, NULL, 241, 1, 4, 126, 5, 90000000, 206200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1216, N'The Incredible Hulk', CAST(N'2008-06-08T00:00:00.000' AS DateTime), 376, 53, NULL, 241, 1, 4, 112, 4, 150000000, 263400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1217, N'Now You See Me', CAST(N'2013-05-21T00:00:00.000' AS DateTime), 376, 108, NULL, 241, 1, 18, 115, 3, 75000000, 351700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1218, N'Clash of the Titans', CAST(N'2010-04-02T00:00:00.000' AS DateTime), 376, 43, NULL, 241, 1, 8, 106, 4, 125000000, 493200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1219, N'Clash of the Titans', CAST(N'1981-06-12T00:00:00.000' AS DateTime), 382, 11, NULL, 240, 1, 8, 118, 3, 15000000, 41000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1220, N'The Transporter', CAST(N'2002-10-02T00:00:00.000' AS DateTime), 376, 196, NULL, 79, 1, 4, 92, 5, 24000000, 43900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1221, N'Transporter 2', CAST(N'2005-08-03T00:00:00.000' AS DateTime), 376, 196, NULL, 79, 1, 4, 88, 5, 32000000, 85160000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1222, N'Zu Warriors from the Magic Mountain', CAST(N'1983-02-05T00:00:00.000' AS DateTime), 370, 103, NULL, 265, 2, 8, 98, 2, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1223, N'A Better Tomorrow', CAST(N'1986-08-02T00:00:00.000' AS DateTime), 45, 198, NULL, 265, 2, 4, 95, 6, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1224, N'A Better Tomorrow II', CAST(N'1987-12-17T00:00:00.000' AS DateTime), 45, 198, NULL, 265, 2, 4, 104, 6, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1225, N'The Master', CAST(N'1992-05-28T00:00:00.000' AS DateTime), 370, 103, NULL, 265, 2, 17, 92, 6, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1226, N'A Better Tomorrow III', CAST(N'1989-10-20T00:00:00.000' AS DateTime), 370, 199, NULL, 265, 2, 4, 145, 6, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1227, N'The Killer', CAST(N'1989-07-06T00:00:00.000' AS DateTime), 45, 199, NULL, 265, 2, 4, 110, 6, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1228, N'Seven Swords', CAST(N'2005-07-29T00:00:00.000' AS DateTime), 370, 200, NULL, 265, 5, 17, 153, 5, 18000000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1229, N'Detective Dee and the Mystery of the Phantom Flame', CAST(N'2010-09-29T00:00:00.000' AS DateTime), 370, 200, NULL, 48, 5, 8, 122, 3, 20000000, 51700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1230, N'Flying Swords of Dragon Gate', CAST(N'2011-12-15T00:00:00.000' AS DateTime), 370, 200, NULL, 48, 5, 17, 125, 5, 35000000, 100000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1231, N'Young Detective Dee: Rise of the Sea Dragon', CAST(N'2013-09-28T00:00:00.000' AS DateTime), 370, 200, NULL, 48, 5, 8, 133, 3, NULL, 98400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1232, N'Hard Boiled', CAST(N'1992-04-16T00:00:00.000' AS DateTime), 45, 199, NULL, 265, 2, 4, 128, 6, 4500000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1233, N'Hard Target', CAST(N'1993-08-20T00:00:00.000' AS DateTime), 45, 137, NULL, 241, 1, 4, 97, 6, 19500000, 74200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1234, N'Broken Arrow', CAST(N'1996-02-09T00:00:00.000' AS DateTime), 45, 4, NULL, 241, 1, 4, 108, 5, 50000000, 150200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1235, N'Face/Off', CAST(N'1997-06-27T00:00:00.000' AS DateTime), 45, 2, NULL, 241, 1, 4, 139, 6, 80000000, 245700000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1236, N'Paycheck', CAST(N'2003-12-25T00:00:00.000' AS DateTime), 45, 60, NULL, 241, 1, 3, 119, 4, 60000000, 96300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1237, N'Red Cliff', CAST(N'2008-07-10T00:00:00.000' AS DateTime), 45, 134, NULL, 48, 5, 5, 146, 5, 80000000, 124000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1238, N'Red Cliff II', CAST(N'2009-01-07T00:00:00.000' AS DateTime), 45, 134, NULL, 48, 5, 5, 142, 5, 80000000, 124300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1239, N'The Raid', CAST(N'2011-09-08T00:00:00.000' AS DateTime), 384, 201, NULL, 266, 13, 17, 101, 6, 1100000, 9140000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1240, N'Merantau', CAST(N'2009-08-06T00:00:00.000' AS DateTime), 384, 201, NULL, 266, 13, 17, 135, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1241, N'The Raid 2', CAST(N'2014-01-21T00:00:00.000' AS DateTime), 384, 201, NULL, 266, 13, 17, 150, 6, 4500000, 6600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1242, N'Ong-Bak', CAST(N'2003-01-21T00:00:00.000' AS DateTime), 385, 202, NULL, 267, 14, 17, 108, 6, 1100000, 20112926, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1243, N'Ong-Bak 2', CAST(N'2008-12-04T00:00:00.000' AS DateTime), 387, 203, NULL, 267, 14, 17, 98, 5, 8000000, 9000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1244, N'Ong-Bak 3', CAST(N'2010-05-05T00:00:00.000' AS DateTime), 387, 203, NULL, 267, 14, 17, 95, 6, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1245, N'Warrior King', CAST(N'2005-08-11T00:00:00.000' AS DateTime), 385, 202, NULL, 267, 14, 17, 110, 6, 5000000, 27200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1246, N'Warrior King 2', CAST(N'2013-10-23T00:00:00.000' AS DateTime), 385, 202, NULL, 267, 14, 17, 103, 5, 15000000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1247, N'Ip Man', CAST(N'2008-12-18T00:00:00.000' AS DateTime), 388, 134, NULL, 265, 2, 17, 108, 5, 11715000, 21900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1248, N'Ip Man 2', CAST(N'2010-04-29T00:00:00.000' AS DateTime), 388, 134, NULL, 265, 2, 17, 109, 5, 12900000, 49700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1249, N'Ip Man 3', CAST(N'2015-12-16T00:00:00.000' AS DateTime), 388, 204, NULL, 265, 2, 17, 105, 4, 36000000, 156844000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1250, N'Iron Monkey', CAST(N'1993-09-03T00:00:00.000' AS DateTime), 367, 103, NULL, 265, 2, 17, 90, 3, 11000000, 15000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1251, N'Crouching Tiger, Hidden Dragon: Sword of Destiny', CAST(N'2016-02-18T00:00:00.000' AS DateTime), 367, 134, NULL, 241, 1, 17, 105, 5, 20000000, 31000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1252, N'Dragon Tiger Gate', CAST(N'2006-07-28T00:00:00.000' AS DateTime), 388, 134, NULL, 265, 2, 17, 96, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1253, N'Rashomon', CAST(N'1950-08-25T00:00:00.000' AS DateTime), 122, 205, NULL, 118, 6, 2, 88, 2, 250000, NULL, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1254, N'Ikiru', CAST(N'1952-10-09T00:00:00.000' AS DateTime), 122, 38, NULL, 118, 6, 2, 143, 2, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1255, N'Throne of Blood', CAST(N'1957-01-15T00:00:00.000' AS DateTime), 122, 38, NULL, 118, 6, 2, 110, 2, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1256, N'The Hidden Fortress', CAST(N'1958-12-28T00:00:00.000' AS DateTime), 122, 38, NULL, 118, 6, 8, 139, 2, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1257, N'Yojimbo', CAST(N'1961-04-25T00:00:00.000' AS DateTime), 122, 38, NULL, 118, 6, 4, 110, 2, NULL, NULL, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1258, N'Sanjuro', CAST(N'1962-01-01T00:00:00.000' AS DateTime), 122, 38, NULL, 118, 6, 4, 95, 2, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1259, N'47 Ronin', CAST(N'1994-10-22T00:00:00.000' AS DateTime), 389, 38, NULL, 118, 6, 4, 132, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1260, N'47 Samurai', CAST(N'1962-11-03T00:00:00.000' AS DateTime), 390, 38, NULL, 118, 6, 4, 207, 7, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1261, N'47 Ronin', CAST(N'2013-12-06T00:00:00.000' AS DateTime), 391, 206, N'Hilariously bad version of the story where some random white dude samurais better than all of the actual samurai. Yet I can''t help myself watching it whenever it''s on TV.', 241, 1, 4, 127, 4, 225000000, 151800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1262, N'The Tale of Osaka Castle', CAST(N'1961-01-03T00:00:00.000' AS DateTime), 390, 38, NULL, 118, 6, 4, 95, 7, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1263, N'Whirlwind', CAST(N'1964-01-03T00:00:00.000' AS DateTime), 390, 38, NULL, 118, 6, 4, 108, 7, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1264, N'Samurai Saga', CAST(N'1959-04-28T00:00:00.000' AS DateTime), 390, 38, NULL, 118, 6, 4, 111, 7, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1265, N'Nippon tanjo', CAST(N'1959-01-01T00:00:00.000' AS DateTime), 390, 38, NULL, 118, 6, 12, 182, 7, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1266, N'Samurai I: Musashi Miyamoto', CAST(N'1954-09-26T00:00:00.000' AS DateTime), 390, 38, NULL, 118, 6, 4, 93, 2, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1267, N'Samurai II: Duel at Ichijoji Temple', CAST(N'1955-07-12T00:00:00.000' AS DateTime), 390, 38, NULL, 118, 6, 4, 103, 2, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1268, N'Samurai III: Duel at Ganryu Island', CAST(N'1956-01-03T00:00:00.000' AS DateTime), 390, 38, NULL, 118, 6, 4, 105, 2, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1269, N'Ichi the Killer', CAST(N'2001-09-14T00:00:00.000' AS DateTime), 392, 207, NULL, 118, 6, 14, 128, 6, 1400000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1270, N'13 Assassins', CAST(N'2010-09-25T00:00:00.000' AS DateTime), 392, 38, N'Makes the battle scene in The Last Samurai look like a tickling contest.', 118, 6, 4, 125, 6, 6000000, 17100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1271, N'Hara-Kiri: Death of a Samurai', CAST(N'2011-05-19T00:00:00.000' AS DateTime), 392, 208, NULL, 118, 6, 2, 126, 6, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1272, N'Zatoichi', CAST(N'2003-09-02T00:00:00.000' AS DateTime), 393, 209, NULL, 118, 6, 4, 116, 6, NULL, 32300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1273, N'Thor', CAST(N'2011-04-17T00:00:00.000' AS DateTime), 394, 53, NULL, 241, 1, 4, 114, 4, 150000000, 449300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1274, N'Battleship', CAST(N'2012-04-03T00:00:00.000' AS DateTime), 167, 210, NULL, 241, 1, 4, 131, 4, 220000000, 303000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1275, N'The Kingdom', CAST(N'2007-08-22T00:00:00.000' AS DateTime), 167, 128, NULL, 241, 1, 9, 109, 5, 70000000, 119000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1276, N'Lone Survivor', CAST(N'2013-11-12T00:00:00.000' AS DateTime), 167, 82, NULL, 241, 1, 5, 121, 5, 40000000, 154800000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1277, N'Deepwater Horizon', CAST(N'2016-09-13T00:00:00.000' AS DateTime), 167, 46, NULL, 241, 1, 25, 107, 4, 110000000, 117300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1278, N'Boiling Point', CAST(N'1990-09-15T00:00:00.000' AS DateTime), 393, 211, NULL, 118, 6, 14, 96, 6, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1279, N'Outrage', CAST(N'2010-05-17T00:00:00.000' AS DateTime), 393, 211, NULL, 118, 6, 14, 109, 6, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1280, N'Beyond Outrage', CAST(N'2012-09-02T00:00:00.000' AS DateTime), 393, 211, NULL, 118, 6, 14, 112, 7, NULL, 16211000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1281, N'Battle Royale', CAST(N'2000-12-16T00:00:00.000' AS DateTime), 395, 212, NULL, 118, 6, 4, 113, 6, 4500000, 25000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1282, N'The Fall of Ako Castle', CAST(N'1978-10-28T00:00:00.000' AS DateTime), 395, 213, NULL, 118, 6, 4, 158, 7, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1283, N'Shogun''s Samurai', CAST(N'1978-01-21T00:00:00.000' AS DateTime), 395, 213, NULL, 118, 6, 4, 130, 7, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1284, N'Samurai Reincarnation', CAST(N'1981-06-06T00:00:00.000' AS DateTime), 395, 213, NULL, 118, 6, 4, 122, 7, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1285, N'The Fast and the Furious: Tokyo Drift', CAST(N'2006-06-04T00:00:00.000' AS DateTime), 142, 128, NULL, 241, 1, 4, 104, 4, 85000000, 158500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1286, N'The Bushido Blade', CAST(N'1981-11-20T00:00:00.000' AS DateTime), 396, 214, NULL, 241, 1, 4, 104, 7, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1287, N'When the Last Sword Is Drawn', CAST(N'2002-11-04T00:00:00.000' AS DateTime), 397, 215, NULL, 118, 6, 2, 137, 5, 4000000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1288, N'Departures', CAST(N'2008-08-23T00:00:00.000' AS DateTime), 397, 216, NULL, 118, 6, 2, 130, 4, NULL, 70000000, 1, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1289, N'The Twilight Samurai', CAST(N'2002-11-02T00:00:00.000' AS DateTime), 398, 217, NULL, 118, 6, 2, 129, 3, 5000000, NULL, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1290, N'The Hidden Blade', CAST(N'2004-10-30T00:00:00.000' AS DateTime), 398, 217, NULL, 118, 6, 2, 132, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1291, N'Love and Honor', CAST(N'2006-10-20T00:00:00.000' AS DateTime), 398, 209, NULL, 118, 6, 2, 118, 3, NULL, 33755000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1292, N'Lone Wolf and Cub: Sword of Vengeance', CAST(N'1972-04-01T00:00:00.000' AS DateTime), 399, 38, NULL, 118, 6, 8, 83, 6, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1293, N'Lone Wolf and Cub: Baby Cart at the River Styx', CAST(N'1972-04-27T00:00:00.000' AS DateTime), 399, 38, NULL, 118, 6, 8, 85, 6, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1294, N'Lone Wolf and Cub: Baby Cart to Hades', CAST(N'1972-09-02T00:00:00.000' AS DateTime), 399, 38, NULL, 118, 6, 8, 89, 6, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1295, N'Lone Wolf and Cub: Baby Cart in Peril', CAST(N'1972-11-01T00:00:00.000' AS DateTime), 400, 38, NULL, 118, 6, 8, 81, 6, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1296, N'Lone Wolf and Cub: Baby Cart in the Land of Demons', CAST(N'1973-08-11T00:00:00.000' AS DateTime), 399, 38, NULL, 118, 6, 8, 89, 6, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1297, N'Lone Wolf and Cub: White Heaven in Hell', CAST(N'1974-04-24T00:00:00.000' AS DateTime), 401, 38, NULL, 118, 6, 8, 84, 6, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1298, N'Godzilla', CAST(N'1954-11-03T00:00:00.000' AS DateTime), 402, 38, NULL, 118, 6, 4, 96, 2, 175000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1299, N'King Kong vs. Godzilla', CAST(N'1962-08-11T00:00:00.000' AS DateTime), 402, 38, NULL, 118, 6, 4, 97, 2, 200000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1300, N'Mothra vs. Godzilla', CAST(N'1964-04-29T00:00:00.000' AS DateTime), 402, 38, NULL, 118, 6, 4, 88, 2, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1301, N'District 13', CAST(N'2004-11-10T00:00:00.000' AS DateTime), 403, 196, NULL, 79, 3, 4, 86, 5, 12000000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1302, N'District 13: Ultimatum', CAST(N'2009-02-18T00:00:00.000' AS DateTime), 404, 196, NULL, 79, 3, 4, 106, 5, 13600000, 14100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1303, N'Taken', CAST(N'2008-02-27T00:00:00.000' AS DateTime), 403, 196, NULL, 79, 1, 9, 90, 6, 22000000, 226800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1304, N'From Paris with Love', CAST(N'2010-02-05T00:00:00.000' AS DateTime), 403, 196, NULL, 79, 1, 9, 92, 5, 52000000, 52800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1305, N'The Gunman', CAST(N'2015-03-20T00:00:00.000' AS DateTime), 403, 52, NULL, 241, 1, 9, 115, 5, 40000000, 24200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1306, N'Mesrine Part 1: Killer Instinct', CAST(N'2008-10-22T00:00:00.000' AS DateTime), 405, 218, NULL, 79, 3, 14, 113, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1307, N'Mesrine Part 2: Public Enemy Number 1', CAST(N'2008-11-19T00:00:00.000' AS DateTime), 405, 218, NULL, 79, 3, 14, 133, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1308, N'Assault on Precinct 13', CAST(N'2005-01-19T00:00:00.000' AS DateTime), 405, 219, NULL, 241, 1, 9, 109, 5, 30000000, 35300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1309, N'A Prophet', CAST(N'2009-05-16T00:00:00.000' AS DateTime), 406, 220, NULL, 79, 3, 14, 155, 6, 13000000, 17900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1310, N'Kung Fu Hustle', CAST(N'2004-09-14T00:00:00.000' AS DateTime), 182, 134, NULL, 265, 2, 17, 98, 5, 20000000, 102000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1311, N'This Is England', CAST(N'2006-09-12T00:00:00.000' AS DateTime), 407, 221, NULL, 240, 1, 2, 102, 6, 3000000, 10000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1312, N'A Room for Romeo Brass', CAST(N'1999-08-23T00:00:00.000' AS DateTime), 407, 222, NULL, 240, 1, 2, 90, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1313, N'Once Upon a Time in the Midlands', CAST(N'2002-09-06T00:00:00.000' AS DateTime), 407, 221, NULL, 240, 1, 27, 104, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1314, N'Dead Man''s Shoes', CAST(N'2004-10-01T00:00:00.000' AS DateTime), 407, 221, N'Love this film - Paddy Considine is terrifying. And it was filmed just an hour from where I live!', 240, 1, 9, 90, 6, 1000000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1315, N'The League of Gentlemen''s Apocalypse', CAST(N'2005-06-03T00:00:00.000' AS DateTime), 408, 221, N'Filmed in the town in which I live. Which is the only exciting thing that has ever happened there.', 240, 1, 7, 92, 5, 6000000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1316, N'Down Terrace', CAST(N'2009-09-21T00:00:00.000' AS DateTime), 409, 223, NULL, 240, 1, 14, 93, 5, 30000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1317, N'Kill List', CAST(N'2011-03-12T00:00:00.000' AS DateTime), 409, 221, NULL, 240, 1, 9, 95, 6, 800000, 462206, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1318, N'Sightseers', CAST(N'2012-05-23T00:00:00.000' AS DateTime), 409, 221, NULL, 240, 1, 7, 85, 5, NULL, 2101166, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1319, N'A Field in England', CAST(N'2013-07-05T00:00:00.000' AS DateTime), 409, 221, NULL, 240, 1, 9, 91, 5, 500000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1320, N'High-Rise', CAST(N'2015-09-13T00:00:00.000' AS DateTime), 409, 221, NULL, 240, 1, 3, 119, 5, 8000000, 3900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1321, N'Free Fire', CAST(N'2016-09-08T00:00:00.000' AS DateTime), 409, 221, NULL, 240, 1, 7, 90, 7, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1322, N'Momentum', CAST(N'2015-10-16T00:00:00.000' AS DateTime), 410, 224, NULL, 268, 1, 9, 96, 5, 20000000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1323, N'District 9', CAST(N'2009-08-13T00:00:00.000' AS DateTime), 411, 42, NULL, 268, 1, 3, 112, 5, 30000000, 210800000, 4, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1324, N'Elysium', CAST(N'2013-08-07T00:00:00.000' AS DateTime), 411, 147, NULL, 241, 1, 3, 109, 5, 115000000, 286100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1325, N'Chappie', CAST(N'2015-03-04T00:00:00.000' AS DateTime), 411, 5, NULL, 241, 1, 3, 120, 5, 49000000, 102100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1326, N'Slumdog Millionaire', CAST(N'2008-08-30T00:00:00.000' AS DateTime), 108, 221, NULL, 240, 1, 2, 120, 5, 15000000, 377900000, 10, 8);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1327, N'The Best Exotic Marigold Hotel', CAST(N'2011-11-30T00:00:00.000' AS DateTime), 195, 225, NULL, 240, 1, 7, 124, 4, 10000000, 136800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1328, N'The Second Best Exotic Marigold Hotel', CAST(N'2015-02-26T00:00:00.000' AS DateTime), 195, 225, NULL, 240, 1, 7, 122, 2, 10000000, 86000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1329, N'John Carter', CAST(N'2012-03-07T00:00:00.000' AS DateTime), 170, 3, NULL, 241, 1, 3, 132, 4, 350000000, 284100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1330, N'Ironclad', CAST(N'2011-03-04T00:00:00.000' AS DateTime), 412, 226, NULL, 240, 1, 13, 121, 5, 25000000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1331, N'Solomon Kane', CAST(N'2009-12-23T00:00:00.000' AS DateTime), 413, 68, NULL, 79, 1, 8, 104, 5, 40000000, 19400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1332, N'A Knight''s Tale', CAST(N'2001-05-11T00:00:00.000' AS DateTime), 414, 5, NULL, 241, 1, 8, 132, 2, 65000000, 117500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1333, N'Shallow Grave', CAST(N'1994-05-17T00:00:00.000' AS DateTime), 108, 221, NULL, 240, 1, 14, 92, 6, 2500000, 19800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1334, N'A Life Less Ordinary', CAST(N'1997-10-24T00:00:00.000' AS DateTime), 108, 33, NULL, 240, 1, 7, 103, 5, 12000000, 4366722, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1335, N'28 Weeks Later', CAST(N'2007-04-26T00:00:00.000' AS DateTime), 415, 113, NULL, 240, 1, 10, 100, 6, 15000000, 64200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1336, N'127 Hours', CAST(N'2010-09-04T00:00:00.000' AS DateTime), 108, 221, N'Gruesome!', 240, 1, 19, 93, 5, 18000000, 60700000, 6, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1337, N'Trance', CAST(N'2013-03-19T00:00:00.000' AS DateTime), 108, 172, NULL, 240, 1, 14, 101, 5, 20000000, 24300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1338, N'Steve Jobs', CAST(N'2015-09-05T00:00:00.000' AS DateTime), 108, 43, NULL, 241, 1, 19, 122, 5, 30000000, 34400000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1339, N'Life of Pi', CAST(N'2012-09-28T00:00:00.000' AS DateTime), 8, 79, NULL, 241, 1, 8, 127, 2, 120000000, 609000000, 11, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1340, N'Terminator Genisys', CAST(N'2015-06-22T00:00:00.000' AS DateTime), 144, 2, NULL, 241, 1, 3, 126, 4, 155000000, 440600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1341, N'Maggie', CAST(N'2015-04-22T00:00:00.000' AS DateTime), 416, 228, NULL, 241, 1, 10, 95, 5, 4500000, 1400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1342, N'Sabotage', CAST(N'2014-03-19T00:00:00.000' AS DateTime), 181, 229, NULL, 241, 1, 14, 109, 5, 35000000, 18400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1343, N'Escape Plan', CAST(N'2013-10-09T00:00:00.000' AS DateTime), 417, 108, NULL, 241, 1, 4, 115, 5, 50000000, 137300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1344, N'The Last Stand', CAST(N'2013-01-18T00:00:00.000' AS DateTime), 418, 46, NULL, 241, 1, 4, 107, 5, 45000000, 48300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1345, N'Terminator Salvation', CAST(N'2009-05-21T00:00:00.000' AS DateTime), 419, 5, NULL, 241, 1, 3, 115, 4, 200000000, 371400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1346, N'Collateral Damage', CAST(N'2002-02-04T00:00:00.000' AS DateTime), 83, 6, NULL, 241, 1, 4, 108, 5, 85000000, 78300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1347, N'The 6th Day', CAST(N'2000-10-28T00:00:00.000' AS DateTime), 47, 5, NULL, 241, 1, 3, 124, 5, 82000000, 96100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1348, N'End of Days', CAST(N'1999-11-16T00:00:00.000' AS DateTime), 420, 230, NULL, 241, 1, 10, 123, 6, 100000000, 212000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1349, N'Batman & Robin', CAST(N'1997-06-20T00:00:00.000' AS DateTime), 204, 104, NULL, 241, 1, 4, 125, 2, 140000000, 238200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1350, N'Jingle All the Way', CAST(N'1996-11-16T00:00:00.000' AS DateTime), 208, 4, NULL, 241, 1, 26, 89, 2, 75000000, 129800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1351, N'Junior', CAST(N'1994-11-23T00:00:00.000' AS DateTime), 94, 1, NULL, 241, 1, 7, 109, 2, 60000000, 108400000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1352, N'Total Recall', CAST(N'2012-08-03T00:00:00.000' AS DateTime), 49, 128, N'Inferior to the original in every conceivable way. Except for Bryan Cranston. He''s great.', 241, 1, 3, 118, 4, 125000000, 198500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1353, N'True Grit', CAST(N'2010-12-22T00:00:00.000' AS DateTime), 5, 2, NULL, 241, 1, 1, 110, 5, 38000000, 252300000, 10, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1354, N'Red Heat', CAST(N'1988-06-14T00:00:00.000' AS DateTime), 250, 15, NULL, 241, 1, 4, 103, 6, 29000000, 35000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1355, N'Predator', CAST(N'1987-06-12T00:00:00.000' AS DateTime), 58, 52, N'Possibly the greatest amount of testosterone ever committed to film.', 241, 1, 3, 107, 6, 18000000, 98300000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1356, N'Raw Deal', CAST(N'1986-06-06T00:00:00.000' AS DateTime), 421, 96, NULL, 241, 1, 4, 107, 6, 11000000, 18100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1357, N'Commando', CAST(N'1985-10-04T00:00:00.000' AS DateTime), 422, 52, NULL, 241, 1, 4, 90, 6, 10000000, 57500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1358, N'X-Men: The Last Stand', CAST(N'2006-05-25T00:00:00.000' AS DateTime), 57, 53, NULL, 241, 1, 4, 104, 4, 210000000, 459400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1359, N'Red Sonja', CAST(N'1985-07-03T00:00:00.000' AS DateTime), 297, 96, NULL, 269, 1, 12, 89, 5, 17900000, 6900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1360, N'Conan the Destroyer', CAST(N'1984-06-29T00:00:00.000' AS DateTime), 297, 96, NULL, 241, 1, 12, 101, 5, 18000000, 100000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1361, N'Conan the Barbarian', CAST(N'1982-03-16T00:00:00.000' AS DateTime), 423, 96, NULL, 241, 1, 12, 129, 5, 16000000, 130000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1362, N'Conan the Barbarian', CAST(N'2011-08-11T00:00:00.000' AS DateTime), 424, 80, NULL, 241, 1, 12, 113, 5, 90000000, 48800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1363, N'Pumping Iron', CAST(N'1977-01-18T00:00:00.000' AS DateTime), 425, 231, NULL, 241, 1, 24, 85, 3, 1000000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1364, N'Hercules in New York', CAST(N'1969-12-16T00:00:00.000' AS DateTime), 426, 232, NULL, 241, 1, 7, 91, 2, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1365, N'3 Days to Kill', CAST(N'2014-02-12T00:00:00.000' AS DateTime), 419, 196, NULL, 79, 1, 9, 117, 4, 28000000, 52600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1366, N'This Means War', CAST(N'2012-02-17T00:00:00.000' AS DateTime), 419, 28, NULL, 241, 1, 4, 97, 4, 65000000, 156500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1367, N'We Are Marshall', CAST(N'2006-12-22T00:00:00.000' AS DateTime), 419, 43, NULL, 241, 1, 2, 131, 2, 65000000, 43500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1368, N'X-Men: First Class', CAST(N'2011-05-25T00:00:00.000' AS DateTime), 427, 53, NULL, 241, 1, 4, 132, 4, 160000000, 353600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1369, N'Charlie''s Angels', CAST(N'2000-11-03T00:00:00.000' AS DateTime), 419, 5, NULL, 241, 1, 4, 98, 5, 90000000, 264100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1370, N'Charlie''s Angels: Full Throttle', CAST(N'2003-06-27T00:00:00.000' AS DateTime), 419, 5, NULL, 241, 1, 4, 106, 4, 120000000, 259100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1371, N'Kingsman: The Secret Service', CAST(N'2014-12-13T00:00:00.000' AS DateTime), 427, 48, NULL, 240, 1, 4, 129, 5, 94000000, 414400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1372, N'Kick-Ass', CAST(N'2010-03-26T00:00:00.000' AS DateTime), 427, 234, NULL, 240, 1, 4, 117, 5, 30000000, 96200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1373, N'Stardust', CAST(N'2007-08-10T00:00:00.000' AS DateTime), 427, 235, N'Every Neil Gaiman book needs to be turned into a movie as good as this one.', 240, 1, 12, 127, 2, 88500000, 135600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1374, N'Layer Cake', CAST(N'2004-10-01T00:00:00.000' AS DateTime), 427, 235, NULL, 240, 1, 14, 105, 5, 6500000, 11900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1375, N'Underworld', CAST(N'2003-09-19T00:00:00.000' AS DateTime), 49, 91, NULL, 240, 1, 4, 121, 5, 22000000, 95700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1376, N'Underworld: Evolution', CAST(N'2006-01-20T00:00:00.000' AS DateTime), 49, 91, NULL, 240, 1, 4, 106, 6, 45000000, 111300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1377, N'For a Few Dollars More', CAST(N'1965-11-18T00:00:00.000' AS DateTime), 308, 34, NULL, 264, 12, 1, 132, 5, 600000, 15000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1378, N'Once Upon a Time in the West', CAST(N'1968-12-21T00:00:00.000' AS DateTime), 308, 2, NULL, 264, 12, 1, 165, 5, 5000000, 5321500, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1379, N'Once Upon a Time in America', CAST(N'1984-05-23T00:00:00.000' AS DateTime), 308, 94, NULL, 241, 1, 14, 229, 6, 30000000, 5300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1380, N'Underworld: Rise of the Lycans', CAST(N'2009-01-23T00:00:00.000' AS DateTime), 428, 91, NULL, 241, 1, 4, 92, 6, 35000000, 91300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1381, N'Underworld: Awakening', CAST(N'2012-01-20T00:00:00.000' AS DateTime), 429, 91, NULL, 241, 1, 4, 89, 6, 70000000, 160000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1382, N'Underworld: Blood Wars', CAST(N'2016-11-24T00:00:00.000' AS DateTime), 430, 91, NULL, 241, 1, 4, 91, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1383, N'Stargate', CAST(N'1994-10-28T00:00:00.000' AS DateTime), 42, 195, NULL, 241, 1, 3, 128, 2, 55000000, 196600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1384, N'Under Siege', CAST(N'1992-10-09T00:00:00.000' AS DateTime), 83, 65, NULL, 241, 1, 4, 103, 5, 35000000, 156600000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1385, N'Bridge of Spies', CAST(N'2015-10-04T00:00:00.000' AS DateTime), 4, 8, NULL, 241, 1, 9, 141, 4, 40000000, 165500000, 6, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1386, N'Duel', CAST(N'1971-11-13T00:00:00.000' AS DateTime), 4, 1, NULL, 241, 1, 9, 89, 2, 450000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1387, N'Empire of the Sun', CAST(N'1987-12-11T00:00:00.000' AS DateTime), 4, 41, NULL, 241, 1, 5, 154, 2, 35000000, 66700000, 6, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1388, N'Amistad', CAST(N'1997-12-10T00:00:00.000' AS DateTime), 4, 8, NULL, 241, 1, 13, 154, 5, 36000000, 44200000, 4, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1389, N'A.I. Artificial Intelligence', CAST(N'2001-06-29T00:00:00.000' AS DateTime), 4, 41, NULL, 241, 1, 3, 146, 3, 100000000, 235900000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1390, N'Catch Me If You Can', CAST(N'2002-12-25T00:00:00.000' AS DateTime), 4, 41, NULL, 241, 1, 19, 141, 4, 52000000, 352100000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1391, N'The Terminal', CAST(N'2004-06-18T00:00:00.000' AS DateTime), 4, 41, NULL, 241, 1, 7, 128, 4, 60000000, 219400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1392, N'Lincoln', CAST(N'2012-10-08T00:00:00.000' AS DateTime), 4, 41, N'I preferred the one where Abe was a vampire hunter.', 241, 1, 19, 150, 4, 65000000, 275300000, 12, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1393, N'Abraham Lincoln: Vampire Hunter', CAST(N'2012-06-18T00:00:00.000' AS DateTime), 119, 236, N'True story.', 241, 1, 13, 105, 5, 99500000, 116400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1394, N'The BFG', CAST(N'2016-05-14T00:00:00.000' AS DateTime), 4, 3, NULL, 241, 1, 26, 117, 2, 140000000, 178000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1395, N'Day Watch', CAST(N'2006-01-01T00:00:00.000' AS DateTime), 119, 109, NULL, 258, 9, 12, 131, 5, 4200000, 38860000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1396, N'Unbroken', CAST(N'2014-11-17T00:00:00.000' AS DateTime), 431, 43, NULL, 241, 1, 19, 137, 5, 65000000, 163400000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1397, N'Inside Llewyn Davis', CAST(N'2013-05-19T00:00:00.000' AS DateTime), 5, 195, NULL, 241, 1, 2, 105, 5, 11000000, 32900000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1398, N'Burn After Reading', CAST(N'2008-08-27T00:00:00.000' AS DateTime), 5, 195, NULL, 241, 1, 7, 96, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1399, N'No Country for Old Men', CAST(N'2007-05-19T00:00:00.000' AS DateTime), 5, 16, NULL, NULL, 1, 14, 122, 5, 25000000, 171600000, 8, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1400, N'The Ladykillers', CAST(N'2004-03-26T00:00:00.000' AS DateTime), 5, 7, NULL, 241, 1, 7, 104, 5, 35000000, 76700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1401, N'The Ladykillers', CAST(N'1955-12-08T00:00:00.000' AS DateTime), 432, 237, NULL, 240, 1, 7, 97, 1, NULL, NULL, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1402, N'The Big Lebowski', CAST(N'1998-03-06T00:00:00.000' AS DateTime), 5, 104, NULL, 241, 1, 7, 117, 6, 15000000, 46200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1403, N'Miller''s Crossing', CAST(N'1990-09-21T00:00:00.000' AS DateTime), 5, 4, NULL, 241, 1, 14, 115, 6, 14000000, 5080000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1404, N'Outland', CAST(N'1981-05-22T00:00:00.000' AS DateTime), 420, 94, NULL, 240, 1, 3, 109, 5, 16000000, 20000000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1405, N'Outlander', CAST(N'2008-07-11T00:00:00.000' AS DateTime), 433, 90, NULL, 241, 1, 3, 115, 5, 47000000, 7033000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1406, N'Timecop', CAST(N'1994-09-16T00:00:00.000' AS DateTime), 420, 142, NULL, 241, 1, 3, 98, 6, 27000000, 101646581, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1407, N'Kung Fu Panda 3', CAST(N'2016-01-23T00:00:00.000' AS DateTime), 154, 8, NULL, 241, 1, 11, 95, 2, 145000000, 519900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1408, N'Universal Soldier', CAST(N'1992-07-10T00:00:00.000' AS DateTime), 42, 15, NULL, 241, 1, 3, 102, 6, 23000000, 36300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1409, N'Double Impact', CAST(N'1991-08-09T00:00:00.000' AS DateTime), 435, 5, NULL, 241, 1, 4, 107, 6, 18000000, 80500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1410, N'Death Warrant', CAST(N'1990-09-14T00:00:00.000' AS DateTime), 436, 172, NULL, 241, 1, 4, 89, 6, 6000000, 46700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1411, N'Kickboxer', CAST(N'1989-09-08T00:00:00.000' AS DateTime), 437, 238, NULL, 241, 1, 17, 103, 6, 2700000, 14700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1412, N'Cyborg', CAST(N'1989-04-07T00:00:00.000' AS DateTime), 438, 239, NULL, 241, 1, 3, 82, 6, 500000, 10200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1413, N'Bloodsport', CAST(N'1988-02-26T00:00:00.000' AS DateTime), 439, 239, NULL, 241, 1, 17, 92, 6, 2300000, 65000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1414, N'Brooklyn''s Finest', CAST(N'2009-01-16T00:00:00.000' AS DateTime), 440, 80, NULL, 241, 1, 14, 132, 6, 17000000, 36400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1415, N'Full Metal Jacket', CAST(N'1987-06-17T00:00:00.000' AS DateTime), 292, 240, NULL, 240, 1, 5, 116, 5, 30000000, 46400000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1416, N'Eyes Wide Shut', CAST(N'1999-07-16T00:00:00.000' AS DateTime), 292, 240, NULL, 240, 1, 2, 159, 6, 65000000, 162100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1417, N'The Shining', CAST(N'1980-05-23T00:00:00.000' AS DateTime), 292, 168, NULL, 240, 1, 10, 146, 6, 19000000, 44400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1418, N'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb', CAST(N'1964-01-29T00:00:00.000' AS DateTime), 292, 168, N'Gentlemen! You can''t fight in here; this is the war room.', 240, 1, 7, 94, 2, 1800000, 9400000, 4, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1419, N'Paths of Glory', CAST(N'1957-12-25T00:00:00.000' AS DateTime), 292, 173, NULL, 241, 1, 5, 88, 2, 900000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1420, N'The Treasure of the Sierra Madre', CAST(N'1948-01-06T00:00:00.000' AS DateTime), 252, 6, NULL, 241, 1, 1, 126, 2, 3000000, 4307000, 4, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1421, N'Key Largo', CAST(N'1948-07-16T00:00:00.000' AS DateTime), 252, 6, NULL, 241, 1, 14, 101, 2, NULL, 3250000, 1, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1422, N'The Unforgiven', CAST(N'1960-04-06T00:00:00.000' AS DateTime), 252, 22, NULL, 241, 1, 1, 125, 2, 5000000, 3200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1423, N'Unforgiven', CAST(N'1992-08-03T00:00:00.000' AS DateTime), 10, 31, NULL, 241, 1, 1, 131, 5, 14400000, 159200000, 9, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1424, N'The Big Sleep', CAST(N'1946-08-23T00:00:00.000' AS DateTime), 328, 6, NULL, 241, 1, 14, 116, 2, NULL, 3000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1425, N'A Fistful of Dynamite', CAST(N'1971-10-29T00:00:00.000' AS DateTime), 308, 22, NULL, 264, 12, 1, 157, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1426, N'To Have and Have Not', CAST(N'1944-10-11T00:00:00.000' AS DateTime), 328, 6, NULL, 241, 1, 6, 100, 2, NULL, 3650000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1427, N'Murder on the Orient Express', CAST(N'1974-11-24T00:00:00.000' AS DateTime), 277, 164, NULL, 240, 1, 14, 131, 2, 2500000, 30000000, 6, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1428, N'The Shootist', CAST(N'1976-08-20T00:00:00.000' AS DateTime), 291, 96, NULL, 241, 1, 1, 100, 2, NULL, 13406000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1429, N'Flash Gordon', CAST(N'1980-12-05T00:00:00.000' AS DateTime), 441, 96, NULL, 240, 1, 3, 114, 2, 20000000, 27100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1430, N'Unforgiven', CAST(N'2013-09-06T00:00:00.000' AS DateTime), 442, 241, NULL, 118, 6, 1, 135, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1431, N'Hang ''Em High', CAST(N'1968-08-03T00:00:00.000' AS DateTime), 443, 31, NULL, 241, 1, 1, 114, 6, 1600000, 6800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1432, N'Coogan''s Bluff', CAST(N'1968-10-02T00:00:00.000' AS DateTime), 291, 31, NULL, 241, 1, 4, 93, 5, 1500000, 3110000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1433, N'Where Eagles Dare', CAST(N'1968-12-04T00:00:00.000' AS DateTime), 444, 11, NULL, 240, 1, 5, 155, 2, 7700000, 21000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1434, N'Kelly''s Heroes', CAST(N'1970-06-23T00:00:00.000' AS DateTime), 444, 11, NULL, 241, 1, 5, 146, 2, 4000000, 5200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1436, N'The Outlaw Josey Wales', CAST(N'1976-06-30T00:00:00.000' AS DateTime), 10, 31, NULL, 241, 1, 1, 135, 6, 3700000, 31800000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1437, N'Heartbreak Ridge', CAST(N'1986-12-05T00:00:00.000' AS DateTime), 10, 31, NULL, 241, 1, 5, 130, 5, 15000000, 121700000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1438, N'The Dead Pool', CAST(N'1988-07-13T00:00:00.000' AS DateTime), 259, 31, NULL, 241, 1, 9, 91, 6, 31000000, 37903000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1439, N'Space Cowboys', CAST(N'2000-08-01T00:00:00.000' AS DateTime), 10, 31, NULL, 241, 1, 9, 130, 2, 60000000, 128900000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1440, N'Gran Torino', CAST(N'2008-12-12T00:00:00.000' AS DateTime), 10, 31, NULL, 241, 1, 2, 116, 5, 33000000, 270000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1441, N'Invictus', CAST(N'2009-12-11T00:00:00.000' AS DateTime), 10, 31, NULL, 241, 1, 22, 133, 4, 60000000, 122200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1442, N'American Sniper', CAST(N'2014-11-11T00:00:00.000' AS DateTime), 10, 49, NULL, 241, 1, 19, 132, 5, 58800000, 547400000, 6, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1443, N'Sully', CAST(N'2016-09-02T00:00:00.000' AS DateTime), 10, 49, N'I was sorely disappointed that this wasn''t a movie about the big blue guy from Monsters Inc.', 241, 1, 19, 96, 4, 60000000, 200700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1444, N'J. Edgar', CAST(N'2011-11-03T00:00:00.000' AS DateTime), 10, 20, NULL, 241, 1, 19, 137, 5, 35000000, 84600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1445, N'Changeling', CAST(N'2008-10-24T00:00:00.000' AS DateTime), 10, 20, NULL, 241, 1, 19, 141, 5, 55000000, 113000000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1446, N'Blade: Trinity', CAST(N'2004-12-08T00:00:00.000' AS DateTime), 445, 53, NULL, 241, 1, 4, 113, 5, 65000000, 128900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1447, N'U.S. Marshals', CAST(N'1998-03-06T00:00:00.000' AS DateTime), 77, 140, NULL, 241, 1, 9, 131, 5, 45000000, 102400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1448, N'Drop Zone', CAST(N'1994-12-09T00:00:00.000' AS DateTime), 236, 2, NULL, 241, 1, 4, 101, 5, 45000000, 28735000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1449, N'Demolition Man', CAST(N'1993-10-08T00:00:00.000' AS DateTime), 446, 52, NULL, 241, 1, 4, 115, 5, 57000000, 159100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1450, N'Passenger 57', CAST(N'1992-11-06T00:00:00.000' AS DateTime), 447, 6, NULL, 241, 1, 4, 84, 5, 15000000, 44065000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1451, N'The Princess Bride', CAST(N'1987-09-25T00:00:00.000' AS DateTime), 217, 242, N'Love, revenge, giants and Sicilians. What''s not to like?', 241, 1, 12, 98, 2, 16000000, 30900000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1452, N'The Long Riders', CAST(N'1980-05-16T00:00:00.000' AS DateTime), 250, 22, NULL, 241, 1, 1, 99, 6, 8000000, 15795000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1453, N'The Wild Bunch', CAST(N'1969-06-18T00:00:00.000' AS DateTime), 448, 181, NULL, 241, 1, 1, 143, 6, 6000000, 11138000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1454, N'American History X', CAST(N'1998-10-30T00:00:00.000' AS DateTime), 449, 14, NULL, 241, 1, 2, 119, 6, 20000000, 23900000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1455, N'Convoy', CAST(N'1978-06-28T00:00:00.000' AS DateTime), 448, 164, NULL, 241, 1, 4, 112, 5, 12000000, 45000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1456, N'Cross of Iron', CAST(N'1977-01-28T00:00:00.000' AS DateTime), 448, 164, NULL, 240, 1, 5, 133, 6, 6000000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1457, N'The Getaway', CAST(N'1972-12-13T00:00:00.000' AS DateTime), 448, 243, NULL, 241, 1, 14, 122, 6, 3350000, 36734000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1458, N'Straw Dogs', CAST(N'1971-12-29T00:00:00.000' AS DateTime), 448, 244, NULL, 241, 1, 9, 117, 6, 2200000, 8000000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1459, N'Judgment at Nuremberg', CAST(N'1961-12-19T00:00:00.000' AS DateTime), 316, 22, NULL, 241, 1, 2, 179, 2, 3000000, 10000000, 11, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1460, N'The Great Escape', CAST(N'1963-06-20T00:00:00.000' AS DateTime), 331, 167, NULL, 241, 1, 5, 172, 2, 3800000, 11744000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1461, N'Zulu', CAST(N'1964-01-22T00:00:00.000' AS DateTime), 450, 2, NULL, 240, 1, 13, 139, 2, 3500000, 8000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1462, N'Lucky Number Slevin', CAST(N'2006-02-24T00:00:00.000' AS DateTime), 451, 11, NULL, 257, 1, 14, 110, 6, 27000000, 56300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1463, N'Star Trek Beyond', CAST(N'2016-07-07T00:00:00.000' AS DateTime), 142, 126, NULL, 241, 1, 3, 122, 4, 185000000, 343500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1464, N'Star Trek Into Darkness', CAST(N'2013-04-23T00:00:00.000' AS DateTime), 44, 126, NULL, 241, 1, 3, 133, 3, 185000000, 467400000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1465, N'Star Trek', CAST(N'2009-04-07T00:00:00.000' AS DateTime), 44, 126, NULL, 241, 1, 3, 127, 3, 150000000, 385700000, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1466, N'RoboCop', CAST(N'1987-07-17T00:00:00.000' AS DateTime), 72, 151, NULL, 241, 1, 3, 101, 6, 13000000, 53400000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1467, N'RoboCop', CAST(N'2014-02-12T00:00:00.000' AS DateTime), 452, 11, NULL, 241, 1, 3, 118, 4, 100000000, 242700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1468, N'Clerks', CAST(N'1994-10-19T00:00:00.000' AS DateTime), 453, 245, NULL, 241, 1, 7, 92, 6, 230000, 3200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1469, N'Mallrats', CAST(N'1995-10-20T00:00:00.000' AS DateTime), 453, 245, NULL, 241, 1, 7, 94, 6, 6100000, 22100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1470, N'Chasing Amy', CAST(N'1997-04-04T00:00:00.000' AS DateTime), 453, 245, NULL, 241, 1, 27, 113, 6, 250000, 12000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1471, N'Dogma', CAST(N'1999-05-21T00:00:00.000' AS DateTime), 453, 245, NULL, 241, 1, 7, 128, 5, 10000000, 31400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1472, N'Bronson', CAST(N'2008-10-17T00:00:00.000' AS DateTime), 454, 246, NULL, 240, 1, 14, 92, 6, 230000, 2300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1473, N'Drive', CAST(N'2011-05-20T00:00:00.000' AS DateTime), 454, 247, NULL, 241, 1, 14, 100, 6, 1500000, 78100000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1474, N'Only God Forgives', CAST(N'2013-05-22T00:00:00.000' AS DateTime), 454, 18, NULL, 270, 1, 9, 90, 6, 4800000, 10300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1475, N'Taxi Driver', CAST(N'1976-02-08T00:00:00.000' AS DateTime), 9, 5, NULL, 241, 1, 2, 113, 6, 1300000, 28300000, 4, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1476, N'Mean Streets', CAST(N'1973-10-02T00:00:00.000' AS DateTime), 9, 6, NULL, 241, 1, 14, 112, 6, 500000, 3000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1477, N'Cape Fear', CAST(N'1962-04-12T00:00:00.000' AS DateTime), 321, 1, NULL, 241, 1, 9, 105, 5, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1478, N'New York, New York', CAST(N'1977-06-21T00:00:00.000' AS DateTime), 9, 22, NULL, 241, 1, 20, 163, 2, 14000000, 13800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1479, N'The King of Comedy', CAST(N'1982-12-18T00:00:00.000' AS DateTime), 9, 248, NULL, 241, 1, 7, 109, 2, 19000000, 2500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1480, N'The Color of Money', CAST(N'1986-10-17T00:00:00.000' AS DateTime), 9, 7, NULL, 241, 1, 2, 120, 5, 13800000, 52200000, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1481, N'The Last Temptation of Christ', CAST(N'1988-08-12T00:00:00.000' AS DateTime), 9, 1, NULL, 241, 1, 2, 162, 6, 7000000, 8900000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1482, N'The Age of Innocence', CAST(N'1993-09-17T00:00:00.000' AS DateTime), 9, 5, NULL, 241, 1, 6, 139, 1, 34000000, 32300000, 5, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1483, N'Silence', CAST(N'2016-11-29T00:00:00.000' AS DateTime), 9, 2, NULL, 241, 1, 2, 159, 5, 50000000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1484, N'The Wolf of Wall Street', CAST(N'2013-12-17T00:00:00.000' AS DateTime), 9, 2, NULL, 241, 1, 19, 179, 6, 155000000, 392000000, 5, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1485, N'Hugo', CAST(N'2011-10-10T00:00:00.000' AS DateTime), 9, 2, NULL, 240, 1, 8, 126, 1, 170000000, 185800000, 11, 5);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1486, N'Shutter Island', CAST(N'2010-02-19T00:00:00.000' AS DateTime), 9, 2, NULL, 241, 1, 9, 138, 5, 80000000, 294800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1487, N'Zoolander', CAST(N'2001-09-28T00:00:00.000' AS DateTime), 455, 49, NULL, 241, 1, 7, 89, 3, 28000000, 60800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1488, N'Labyrinth', CAST(N'1986-06-27T00:00:00.000' AS DateTime), 456, 161, NULL, 240, 1, 12, 101, 1, 25000000, 12900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1489, N'Willow', CAST(N'1988-05-20T00:00:00.000' AS DateTime), 40, 13, NULL, 241, 1, 12, 126, 2, 35000000, 57300000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1490, N'Mad Max', CAST(N'1979-04-12T00:00:00.000' AS DateTime), 173, 154, NULL, 261, 1, 4, 93, 6, 400000, 100000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1491, N'Mad Max 2', CAST(N'1981-12-24T00:00:00.000' AS DateTime), 173, 154, NULL, 261, 1, 4, 96, 6, 4500000, 23700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1492, N'Mad Max Beyond Thunderdome', CAST(N'1985-07-10T00:00:00.000' AS DateTime), 173, 154, NULL, 261, 1, 4, 107, 5, 10000000, 36200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1493, N'Mad Max: Fury Road', CAST(N'2015-05-07T00:00:00.000' AS DateTime), 173, 154, NULL, 261, 1, 4, 120, 5, 150000000, 378400000, 10, 6);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1494, N'Happy Feet Two', CAST(N'2011-11-18T00:00:00.000' AS DateTime), 173, 49, NULL, 241, 1, 11, 100, 1, 135000000, 150400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1495, N'Zoolander 2', CAST(N'2016-02-04T00:00:00.000' AS DateTime), 455, 2, NULL, 241, 1, 7, 102, 4, 50000000, 56000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1496, N'Dune', CAST(N'1984-12-14T00:00:00.000' AS DateTime), 457, 96, NULL, 241, 1, 3, 136, 2, 40000000, 30900000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1497, N'Gods of Egypt', CAST(N'2016-02-25T00:00:00.000' AS DateTime), 53, 108, NULL, 241, 1, 12, 127, 4, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1498, N'The Nice Guys', CAST(N'2016-05-15T00:00:00.000' AS DateTime), 140, 52, NULL, 241, 1, 7, 116, 5, 50000000, 57300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1499, N'Kiss Kiss Bang Bang', CAST(N'2005-05-14T00:00:00.000' AS DateTime), 140, 52, NULL, 241, 1, 7, 103, 5, 15000000, 15800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1500, N'Independence Day: Resurgence', CAST(N'2016-06-20T00:00:00.000' AS DateTime), 42, 57, NULL, 241, 1, 3, 120, 4, 165000000, 389700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1501, N'Ghostbusters', CAST(N'2016-07-09T00:00:00.000' AS DateTime), 458, 49, NULL, 241, 1, 7, 116, 4, 144000000, 229100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1503, N'Jason Bourne', CAST(N'2016-07-11T00:00:00.000' AS DateTime), 37, 56, NULL, 241, 1, 9, 123, 4, 120000000, 415200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1504, N'Sausage Party', CAST(N'2016-03-14T00:00:00.000' AS DateTime), 459, 5, NULL, 241, 1, 11, 88, 5, 19000000, 140200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1505, N'Scott Pilgrim vs. the World', CAST(N'2010-07-27T00:00:00.000' AS DateTime), 106, 30, NULL, 240, 1, 7, 112, 4, 90000000, 47700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1506, N'Death Proof', CAST(N'2007-04-06T00:00:00.000' AS DateTime), 79, 249, NULL, 241, 1, 9, 114, 6, NULL, 30700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1507, N'Jackie Brown', CAST(N'1997-12-25T00:00:00.000' AS DateTime), 79, 84, NULL, 241, 1, 14, 154, 5, 12000000, 74700000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1508, N'Inglourious Basterds', CAST(N'2009-05-20T00:00:00.000' AS DateTime), 79, 84, NULL, 241, 1, 5, 153, 6, 70000000, 321500000, 8, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1509, N'Django Unchained', CAST(N'2012-12-11T00:00:00.000' AS DateTime), 79, 5, NULL, 241, 1, 1, 165, 6, 100000000, 425400000, 5, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1510, N'The Hateful Eight', CAST(N'2015-12-07T00:00:00.000' AS DateTime), 79, 250, NULL, 241, 1, 1, 168, 6, 54000000, 155800000, 3, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1511, N'Bill & Ted''s Excellent Adventure', CAST(N'1989-02-17T00:00:00.000' AS DateTime), 200, 141, NULL, 241, 1, 8, 89, 2, 10000000, 40500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1512, N'Bill & Ted''s Bogus Journey', CAST(N'1991-07-19T00:00:00.000' AS DateTime), 460, 141, NULL, 241, 1, 8, 93, 2, 20000000, 38000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1513, N'The Magnificent Seven', CAST(N'1960-10-23T00:00:00.000' AS DateTime), 331, 167, NULL, 241, 1, 1, 128, 1, 2000000, 2250000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1514, N'The Magnificent Seven', CAST(N'2016-09-08T00:00:00.000' AS DateTime), 440, 49, NULL, 241, 1, 1, 133, 4, 107600000, 161000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1515, N'The Replacement Killers', CAST(N'1998-02-06T00:00:00.000' AS DateTime), 440, 5, NULL, 241, 1, 9, 87, 6, 30000000, 19200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1516, N'Training Day', CAST(N'2001-10-05T00:00:00.000' AS DateTime), 440, 49, NULL, 241, 1, 14, 122, 6, 45000000, 104900000, 2, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1517, N'Tears of the Sun', CAST(N'2003-03-07T00:00:00.000' AS DateTime), 440, 19, NULL, 241, 1, 5, 121, 5, 75000000, 86500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1518, N'King Arthur', CAST(N'2004-07-07T00:00:00.000' AS DateTime), 440, 7, NULL, 241, 1, 8, 142, 4, 120000000, 203600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1519, N'Shooter', CAST(N'2007-03-23T00:00:00.000' AS DateTime), 440, 46, NULL, 241, 1, 4, 125, 5, 61000000, 95700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1520, N'Olympus Has Fallen', CAST(N'2013-03-20T00:00:00.000' AS DateTime), 440, 80, NULL, 241, 1, 4, 119, 5, 70000000, 161000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1521, N'The Equalizer', CAST(N'2014-09-07T00:00:00.000' AS DateTime), 440, 49, NULL, 241, 1, 9, 132, 5, 55000000, 192300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1522, N'Southpaw', CAST(N'2015-06-15T00:00:00.000' AS DateTime), 440, 251, NULL, 241, 1, 2, 123, 5, 30000000, 92000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1523, N'Miss Peregrine''s Home for Peculiar Children', CAST(N'2016-09-25T00:00:00.000' AS DateTime), 28, 236, NULL, 241, 1, 12, 127, 4, 110000000, 271900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1524, N'Mars Attacks!', CAST(N'1996-12-13T00:00:00.000' AS DateTime), 28, 236, NULL, 241, 1, 3, 106, 3, 70000000, 101400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1525, N'Planet of the Apes', CAST(N'2001-07-27T00:00:00.000' AS DateTime), 28, 73, NULL, 241, 1, 3, 120, 3, 100000000, 362200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1526, N'Big Fish', CAST(N'2003-12-10T00:00:00.000' AS DateTime), 28, 73, NULL, 241, 1, 12, 125, 2, 70000000, 122900000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1527, N'Dark Shadows', CAST(N'2012-05-11T00:00:00.000' AS DateTime), 28, 49, NULL, 241, 1, 7, 113, 4, 150000000, 245500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1528, N'Watchmen', CAST(N'2009-02-23T00:00:00.000' AS DateTime), 76, 43, NULL, 241, 1, 4, 163, 6, 130000000, 185300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1529, N'Man of Steel', CAST(N'2013-06-10T00:00:00.000' AS DateTime), 76, 43, NULL, 241, 1, 4, 143, 4, 225000000, 668000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1530, N'Sucker Punch', CAST(N'2011-03-25T00:00:00.000' AS DateTime), 76, 43, NULL, 241, 1, 4, 109, 4, 82000000, 89800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1531, N'Dawn of the Dead', CAST(N'2004-03-19T00:00:00.000' AS DateTime), 76, 74, NULL, 241, 1, 10, 100, 6, 26000000, 102400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1532, N'Jack Reacher: Never Go Back', CAST(N'2016-10-16T00:00:00.000' AS DateTime), 43, 2, NULL, 241, 1, 9, 118, 4, 60000000, 156200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1533, N'Jack Reacher', CAST(N'2012-12-21T00:00:00.000' AS DateTime), 146, 2, NULL, 241, 1, 9, 130, 5, 60000000, 218300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1534, N'The Way of the Gun', CAST(N'2000-09-08T00:00:00.000' AS DateTime), 146, 252, NULL, 241, 1, 4, 119, 6, 8500000, 13069000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1535, N'Fantastic Beasts and Where to Find Them', CAST(N'2016-11-10T00:00:00.000' AS DateTime), 19, 47, NULL, 240, 1, 12, 133, 4, 180000000, 615800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1536, N'Moana', CAST(N'2016-11-14T00:00:00.000' AS DateTime), 215, 3, NULL, 241, 1, 11, 107, 2, 150000000, 190100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1537, N'Bad Santa', CAST(N'2003-11-26T00:00:00.000' AS DateTime), 461, 23, NULL, 241, 1, 7, 91, 5, 23000000, 76500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1538, N'Bad Santa 2', CAST(N'2016-11-15T00:00:00.000' AS DateTime), 462, 16, NULL, 241, 1, 7, 92, 5, 26000000, 22800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1539, N'Rogue One: A Star Wars Story', CAST(N'2016-12-10T00:00:00.000' AS DateTime), 463, 13, NULL, 241, 1, 3, 133, 4, 200000000, 388100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1540, N'Passengers', CAST(N'2016-12-21T00:00:00.000' AS DateTime), 464, 49, NULL, 241, 1, 3, 116, 4, 110000000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1541, N'The Imitation Game', CAST(N'2014-08-29T00:00:00.000' AS DateTime), 464, 253, NULL, 241, 1, 19, 114, 4, 14000000, 233600000, 8, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1542, N'Ex Machina', CAST(N'2015-01-21T00:00:00.000' AS DateTime), 465, 221, NULL, 240, 1, 3, 108, 5, 15000000, 36900000, 2, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1543, N'Anna Karenina', CAST(N'2012-09-07T00:00:00.000' AS DateTime), 466, 195, NULL, 240, 1, 6, 130, 4, 31000000, 69000000, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1544, N'Anna Karenina', CAST(N'1997-04-04T00:00:00.000' AS DateTime), 467, 25, NULL, 241, 1, 6, 108, 5, NULL, 2200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1545, N'Godzilla', CAST(N'2014-05-08T00:00:00.000' AS DateTime), 463, 43, NULL, 241, 1, 4, 123, 4, 160000000, 529100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1546, N'Chocolat', CAST(N'2000-12-15T00:00:00.000' AS DateTime), 468, 16, NULL, 240, 1, 6, 121, 3, 25000000, 152699000, 5, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1547, N'The English Patient', CAST(N'1996-11-15T00:00:00.000' AS DateTime), 469, 16, NULL, 240, 1, 6, 162, 5, 27000000, 232000000, 12, 9);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1548, N'Near Dark', CAST(N'1987-10-02T00:00:00.000' AS DateTime), 470, 96, NULL, 241, 1, 10, 95, 6, 5000000, 3400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1549, N'Blue Steel', CAST(N'1990-03-16T00:00:00.000' AS DateTime), 470, 11, NULL, 241, 1, 4, 101, 6, NULL, 8200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1550, N'Point Break', CAST(N'1991-07-12T00:00:00.000' AS DateTime), 470, 4, NULL, 241, 1, 4, 122, 6, 24000000, 83500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1551, N'Point Break', CAST(N'2015-12-04T00:00:00.000' AS DateTime), 471, 108, NULL, 241, 1, 4, 114, 4, 105000000, 133700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1552, N'Invincible', CAST(N'2006-08-25T00:00:00.000' AS DateTime), 471, 3, NULL, 241, 1, 22, 104, 2, 30000000, 58500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1553, N'Strange Days', CAST(N'1995-10-13T00:00:00.000' AS DateTime), 470, 70, NULL, 241, 1, 3, 145, 6, 42000000, 8000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1554, N'K-19: The Widowmaker', CAST(N'2002-07-19T00:00:00.000' AS DateTime), 470, 132, NULL, 241, 1, 9, 138, 4, 100000000, 65700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1555, N'The Hurt Locker', CAST(N'2008-09-04T00:00:00.000' AS DateTime), 470, 108, NULL, 241, 1, 5, 131, 5, 15000000, 49200000, 9, 6);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1556, N'Zero Dark Thirty', CAST(N'2012-12-19T00:00:00.000' AS DateTime), 470, 5, NULL, 241, 1, 9, 157, 5, 40000000, 132800000, 5, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1557, N'Lock, Stock and Two Smoking Barrels', CAST(N'1998-08-28T00:00:00.000' AS DateTime), 165, 159, NULL, 240, 1, 14, 106, 6, 1350000, 28100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1558, N'Snatch', CAST(N'2000-08-23T00:00:00.000' AS DateTime), 165, 5, NULL, 240, 1, 14, 102, 6, 10000000, 83600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1559, N'Revolver', CAST(N'2005-09-11T00:00:00.000' AS DateTime), 165, 196, NULL, 240, 1, 14, 110, 5, NULL, 6735000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1560, N'RocknRolla', CAST(N'2008-09-04T00:00:00.000' AS DateTime), 165, 6, NULL, 240, 1, 14, 114, 5, 13000000, 25739000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1561, N'Sherlock Holmes: A Game of Shadows', CAST(N'2011-12-16T00:00:00.000' AS DateTime), 165, 49, NULL, 240, 1, 4, 129, 4, 125000000, 545400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1562, N'The Man from U.N.C.L.E.', CAST(N'2015-08-02T00:00:00.000' AS DateTime), 165, 68, NULL, 241, 1, 4, 116, 4, 75000000, 109800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1563, N'Prometheus', CAST(N'2012-05-30T00:00:00.000' AS DateTime), 51, 64, NULL, 241, 1, 3, 124, 5, 130000000, 403400000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1564, N'Legend', CAST(N'1985-12-13T00:00:00.000' AS DateTime), 51, 248, NULL, 241, 1, 12, 125, 2, 24500000, 15500000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1565, N'Black Rain', CAST(N'1989-09-22T00:00:00.000' AS DateTime), 51, 2, NULL, 241, 1, 9, 125, 6, 30000000, 134200000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1566, N'Thelma & Louise', CAST(N'1991-05-24T00:00:00.000' AS DateTime), 51, 172, NULL, 241, 1, 8, 129, 5, 16500000, 45400000, 6, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1567, N'1492: Conquest of Paradise', CAST(N'1992-10-09T00:00:00.000' AS DateTime), 51, 2, NULL, 241, 1, 8, 142, 5, 47000000, 7200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1568, N'G.I. Jane', CAST(N'1997-08-22T00:00:00.000' AS DateTime), 51, 64, NULL, 241, 1, 5, 124, 5, 50000000, 97100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1569, N'Matchstick Men', CAST(N'2003-09-12T00:00:00.000' AS DateTime), 51, 64, NULL, 241, 1, 14, 116, 4, 62000000, 65600000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1570, N'Body of Lies', CAST(N'2008-10-10T00:00:00.000' AS DateTime), 51, 64, NULL, 241, 1, 4, 128, 5, 70000000, 115100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1571, N'Robin Hood', CAST(N'2010-05-12T00:00:00.000' AS DateTime), 51, 64, NULL, 241, 1, 8, 140, 4, 200000000, 321700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1572, N'The Counsellor', CAST(N'2013-10-03T00:00:00.000' AS DateTime), 51, 64, NULL, 240, 1, 9, 118, 6, 25000000, 71000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1573, N'Exodus: Gods and Kings', CAST(N'2014-12-12T00:00:00.000' AS DateTime), 51, 64, NULL, 241, 1, 8, 150, 4, 145000000, 268200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1574, N'Twilight', CAST(N'2008-11-17T00:00:00.000' AS DateTime), 472, 125, N'The single worst thing to happen in over 100 years of film-making. The only way to salvage this film would be to include Wesley Snipes as Blade.', 241, 1, 28, 121, 4, 37000000, 393616000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1575, N'The Girl with the Dragon Tattoo', CAST(N'2009-02-27T00:00:00.000' AS DateTime), 473, 255, NULL, 271, 15, 9, 153, 6, 13000000, 104395000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1577, N'Dead Man Down', CAST(N'2013-03-08T00:00:00.000' AS DateTime), 473, 54, NULL, 241, 1, 14, 117, 5, 30000000, 18100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1578, N'The Girl Who Played with Fire', CAST(N'2009-09-18T00:00:00.000' AS DateTime), 474, 255, NULL, 271, 15, 9, 129, 5, NULL, 67126000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1579, N'The Girl Who Kicked the Hornets'' Nest', CAST(N'2009-11-27T00:00:00.000' AS DateTime), 474, 255, NULL, 271, 15, 9, 147, 5, 5300000, 43500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1580, N'The Girl with the Dragon Tattoo', CAST(N'2011-12-12T00:00:00.000' AS DateTime), 70, 81, NULL, 241, 1, 9, 158, 6, 90000000, 232600000, 5, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1581, N'John Wick', CAST(N'2014-10-13T00:00:00.000' AS DateTime), 475, 108, NULL, 241, 1, 4, 101, 5, 20000000, 86000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1582, N'The Seventh Seal', CAST(N'1957-02-16T00:00:00.000' AS DateTime), 476, 256, NULL, 271, 15, 2, 96, 2, 150000, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1583, N'The Revenant', CAST(N'2015-12-16T00:00:00.000' AS DateTime), 477, 65, NULL, 241, 1, 8, 156, 5, 135000000, 533000000, 12, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1584, N'Birdman', CAST(N'2014-08-27T00:00:00.000' AS DateTime), 477, 65, NULL, 241, 1, 7, 119, 5, 18000000, 103200000, 9, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1585, N'21 Grams', CAST(N'2003-12-26T00:00:00.000' AS DateTime), 477, 257, NULL, 241, 1, 2, 124, 5, 20000000, 60400000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1586, N'Ted', CAST(N'2012-06-29T00:00:00.000' AS DateTime), 478, 1, NULL, 241, 1, 7, 106, 5, 51000000, 549400000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1587, N'Ted 2', CAST(N'2015-06-24T00:00:00.000' AS DateTime), 478, 1, NULL, 241, 1, 7, 115, 5, 68000000, 216700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1588, N'A Million Ways to Die in the West', CAST(N'2014-05-30T00:00:00.000' AS DateTime), 478, 1, NULL, 241, 1, 7, 116, 5, 40000000, 87200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1589, N'Hansel & Gretel: Witch Hunters', CAST(N'2013-01-17T00:00:00.000' AS DateTime), 479, 2, NULL, 241, 1, 4, 88, 5, 50000000, 226300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1590, N'There Will Be Blood', CAST(N'2007-09-27T00:00:00.000' AS DateTime), 482, 16, NULL, 241, 1, 2, 158, 5, 25000000, 76200000, 8, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1591, N'Punch-Drunk Love', CAST(N'2002-05-19T00:00:00.000' AS DateTime), 482, 14, NULL, 241, 1, 6, 95, 5, 25000000, 24700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1592, N'The Master', CAST(N'2012-09-01T00:00:00.000' AS DateTime), 482, 258, NULL, 241, 1, 2, 137, 5, 32000000, 28300000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1593, N'Inherent Vice', CAST(N'2014-10-04T00:00:00.000' AS DateTime), 482, 6, NULL, 241, 1, 14, 149, 5, 20000000, 14700000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1594, N'Boogie Nights', CAST(N'1997-10-10T00:00:00.000' AS DateTime), 482, 14, NULL, 241, 1, 2, 155, 6, 15000000, 43100000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1595, N'Innerspace', CAST(N'1987-07-01T00:00:00.000' AS DateTime), 245, 41, NULL, 241, 1, 3, 120, 2, NULL, 25900000, 1, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1596, N'Gremlins 2: The New Batch', CAST(N'1990-06-15T00:00:00.000' AS DateTime), 245, 41, NULL, 241, 1, 7, 106, 3, 50000000, 41500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1597, N'The Punisher', CAST(N'2004-04-16T00:00:00.000' AS DateTime), 483, 53, NULL, 241, 1, 4, 123, 6, 33000000, 54700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1599, N'The Theory of Everything', CAST(N'2014-09-07T00:00:00.000' AS DateTime), 484, 1, NULL, 240, 1, 19, 123, 4, 15000000, 123700000, 5, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1600, N'Looper', CAST(N'2012-09-06T00:00:00.000' AS DateTime), 485, 147, NULL, 241, 1, 3, 118, 5, 30000000, 176500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1601, N'12 Years a Slave', CAST(N'2013-08-30T00:00:00.000' AS DateTime), 486, 108, NULL, 241, 1, 19, 134, 5, 22000000, 187700000, 9, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1602, N'Prisoners', CAST(N'2013-08-30T00:00:00.000' AS DateTime), 487, 6, NULL, 241, 1, 9, 153, 5, 46000000, 122100000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1603, N'Cowboys & Aliens', CAST(N'2011-07-29T00:00:00.000' AS DateTime), 160, 8, NULL, 241, 1, 3, 118, 4, 163000000, 174800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1604, N'Knight and Day', CAST(N'2010-06-16T00:00:00.000' AS DateTime), 488, 65, NULL, 241, 1, 4, 110, 4, 125000000, 261900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1605, N'Man on Wire', CAST(N'2008-01-22T00:00:00.000' AS DateTime), 484, 222, N'I feel queasy just thinking about this!', 240, 1, 24, 94, 3, 1900000, 5300000, 1, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1606, N'The Danish Girl', CAST(N'2015-09-05T00:00:00.000' AS DateTime), 489, 260, NULL, 240, 1, 19, 119, 5, 15000000, 64200000, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1607, N'Jupiter Ascending', CAST(N'2015-01-27T00:00:00.000' AS DateTime), 145, 49, NULL, 241, 1, 3, 127, 4, 176000000, 184000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1608, N'Black Death', CAST(N'2010-06-11T00:00:00.000' AS DateTime), 490, 259, NULL, 257, 1, 8, 101, 5, NULL, 272445, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1609, N'The Other Boleyn Girl', CAST(N'2008-02-15T00:00:00.000' AS DateTime), 491, 222, NULL, 240, 1, 2, 115, 4, 35000000, 77700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1610, N'Mandela: Long Walk to Freedom', CAST(N'2013-09-07T00:00:00.000' AS DateTime), 491, 172, NULL, 240, 1, 19, 146, 4, 35000000, 27300000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1612, N'Les Misérables', CAST(N'2012-12-05T00:00:00.000' AS DateTime), 489, 260, NULL, 240, 1, 20, 158, 4, 61000000, 441800000, 8, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1613, N'The King''s Speech', CAST(N'2010-09-06T00:00:00.000' AS DateTime), 489, 260, NULL, 240, 1, 19, 119, 4, 15000000, 414200000, 12, 4);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1614, N'Sicario', CAST(N'2015-05-19T00:00:00.000' AS DateTime), 487, 228, NULL, 241, 1, 9, 121, 5, 30000000, 85000000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1615, N'Edge of Tomorrow', CAST(N'2014-05-28T00:00:00.000' AS DateTime), 52, 49, NULL, 241, 1, 3, 113, 4, 178000000, 370500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1616, N'The Adjustment Bureau', CAST(N'2011-02-14T00:00:00.000' AS DateTime), 492, 1, NULL, 241, 1, 3, 106, 4, 50200000, 127900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1617, N'The Wolfman', CAST(N'2010-01-27T00:00:00.000' AS DateTime), 187, 128, NULL, 241, 1, 10, 103, 5, 150000000, 139800000, 1, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1618, N'Charlie Wilson''s War', CAST(N'2007-12-21T00:00:00.000' AS DateTime), 306, 128, NULL, 241, 1, 7, 100, 5, 75000000, 119000000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1619, N'Sin City: A Dame to Kill For', CAST(N'2014-08-19T00:00:00.000' AS DateTime), 103, 23, NULL, 241, 1, 4, 102, 6, 65000000, 39400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1620, N'Shame', CAST(N'2011-09-04T00:00:00.000' AS DateTime), 486, 221, NULL, 240, 1, 2, 101, 6, 6500000, 17700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1621, N'Ant-Man', CAST(N'2015-06-29T00:00:00.000' AS DateTime), 493, 53, NULL, 241, 1, 4, 117, 4, 130000000, 519300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1622, N'3:10 to Yuma', CAST(N'2007-08-22T00:00:00.000' AS DateTime), 488, 128, NULL, 241, 1, 1, 122, 5, 55000000, 70000000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1623, N'3:10 to Yuma', CAST(N'1957-08-07T00:00:00.000' AS DateTime), 494, 5, NULL, 241, 1, 1, 92, 2, NULL, 1850000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1624, N'The Wolverine', CAST(N'2013-07-16T00:00:00.000' AS DateTime), 488, 53, NULL, 241, 1, 4, 126, 4, 120000000, 414800000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1625, N'Rush', CAST(N'2013-09-02T00:00:00.000' AS DateTime), 40, 20, NULL, 240, 1, 22, 122, 5, 38000000, 98200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1626, N'Solaris', CAST(N'2002-11-29T00:00:00.000' AS DateTime), 48, 70, NULL, 241, 1, 3, 98, 4, 47000000, 30000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1627, N'Magic Mike', CAST(N'2012-06-24T00:00:00.000' AS DateTime), 48, 6, NULL, 241, 1, 2, 110, 5, 7000000, 167200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1628, N'The Three Musketeers', CAST(N'1993-11-12T00:00:00.000' AS DateTime), 200, 3, NULL, 241, 1, 8, 105, 2, 30000000, 53900000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1629, N'Mr. Holland''s Opus', CAST(N'1995-12-29T00:00:00.000' AS DateTime), 200, 104, NULL, 241, 1, 2, 143, 2, 31000000, 106269000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1630, N'The Mighty Ducks', CAST(N'1992-10-02T00:00:00.000' AS DateTime), 200, 3, NULL, 241, 1, 7, 101, 2, 10000000, 50700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1631, N'Young Guns', CAST(N'1988-08-12T00:00:00.000' AS DateTime), 495, 21, NULL, 241, 1, 1, 103, 6, 11000000, 45661000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1632, N'Young Guns II: Blaze of Glory', CAST(N'1990-08-01T00:00:00.000' AS DateTime), 496, 21, NULL, 241, 1, 1, 104, 3, 10000000, 44143000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1633, N'The Lost Boys', CAST(N'1987-07-31T00:00:00.000' AS DateTime), 204, 6, NULL, 241, 1, 10, 98, 5, 8500000, 32200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1634, N'Flatliners', CAST(N'1990-08-10T00:00:00.000' AS DateTime), 204, 5, NULL, 241, 1, 10, 115, 5, 26000000, 61500000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1635, N'21 Jump Street', CAST(N'2012-03-16T00:00:00.000' AS DateTime), 497, 128, NULL, 241, 1, 7, 109, 5, 55000000, 201500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1636, N'Cloudy with a Chance of Meatballs', CAST(N'2009-09-18T00:00:00.000' AS DateTime), 497, 127, NULL, 241, 1, 11, 90, 1, 100000000, 243000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1637, N'Anastasia', CAST(N'1956-12-13T00:00:00.000' AS DateTime), 499, 4, NULL, 241, 1, 2, 105, 1, 3520000, 5000000, 2, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1638, N'Hot Rod', CAST(N'2007-08-03T00:00:00.000' AS DateTime), 500, 2, NULL, 241, 1, 7, 88, 4, 25300000, 14334000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1639, N'Cloudy with a Chance of Meatballs 2', CAST(N'2013-09-27T00:00:00.000' AS DateTime), 501, 127, N'"There''s a leek in my boat!";
Best. Joke. Ever.', 241, 1, 11, 95, 1, 78000000, 274300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1640, N'The Lego Movie', CAST(N'2014-02-07T00:00:00.000' AS DateTime), 497, 49, NULL, 241, 1, 11, 100, 1, 60000000, 469200000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1641, N'Bad Boys', CAST(N'1995-04-07T00:00:00.000' AS DateTime), 18, 17, NULL, 241, 1, 4, 119, 6, 23000000, 141400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1642, N'22 Jump Street', CAST(N'2014-06-13T00:00:00.000' AS DateTime), 497, 54, NULL, 241, 1, 7, 112, 5, 84500000, 331300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1643, N'Ice Age: Collision Course', CAST(N'2016-07-22T00:00:00.000' AS DateTime), 149, 4, NULL, 241, 1, 11, 94, 1, 105000000, 407700000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1644, N'The Eagle', CAST(N'2011-03-25T00:00:00.000' AS DateTime), 502, 221, NULL, 240, 1, 8, 114, 4, 25000000, 35500000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1645, N'The Last King of Scotland', CAST(N'2007-01-12T00:00:00.000' AS DateTime), 502, 221, NULL, 240, 1, 2, 123, 5, 6000000, 48400000, 1, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1646, N'Centurion', CAST(N'2010-04-23T00:00:00.000' AS DateTime), 503, 260, NULL, 240, 1, 8, 97, 5, 12000000, 6168000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1647, N'Jason and the Argonauts', CAST(N'1963-06-19T00:00:00.000' AS DateTime), 504, 5, NULL, 240, 1, 12, 101, 1, 1000000, 2100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1648, N'Pete''s Dragon', CAST(N'1977-11-03T00:00:00.000' AS DateTime), 504, 3, NULL, 241, 1, 12, 134, 1, 10000000, 39600000, 2, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1649, N'Pete''s Dragon', CAST(N'2016-08-12T00:00:00.000' AS DateTime), 505, 3, NULL, 241, 1, 12, 102, 2, 65000000, 142400000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1650, N'One Million Years B.C.', CAST(N'1966-12-30T00:00:00.000' AS DateTime), 504, 261, NULL, 240, 1, 8, 100, 2, 500000, 8000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1651, N'The Three Musketeers', CAST(N'1973-12-11T00:00:00.000' AS DateTime), 313, 4, NULL, 240, 1, 8, 105, 1, NULL, 10100000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1652, N'The Return of the Musketeers', CAST(N'1989-04-19T00:00:00.000' AS DateTime), 313, 1, NULL, 240, 1, 8, 102, 2, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1653, N'Superman III', CAST(N'1983-06-17T00:00:00.000' AS DateTime), 313, 83, NULL, 240, 1, 4, 125, 2, 39000000, 80200000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1654, N'The Four Musketeers', CAST(N'1974-10-31T00:00:00.000' AS DateTime), 313, 4, NULL, 240, 1, 8, 108, 2, NULL, 8760000, 1, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1655, N'Fantastic Voyage', CAST(N'1966-08-24T00:00:00.000' AS DateTime), 297, 4, NULL, 241, 1, 3, 100, 2, 5100000, 12000000, 5, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1656, N'Julius Caesar', CAST(N'1970-06-04T00:00:00.000' AS DateTime), 506, 262, NULL, 240, 1, 13, 116, 1, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1657, N'Othello', CAST(N'1965-12-15T00:00:00.000' AS DateTime), 506, 263, NULL, 240, 1, 2, 165, 1, NULL, NULL, 4, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1658, N'Julius Caesar', CAST(N'1953-06-04T00:00:00.000' AS DateTime), 314, 11, NULL, 241, 1, 13, 121, 2, 2070000, 3920000, 5, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1659, N'Othello', CAST(N'1951-11-29T00:00:00.000' AS DateTime), 352, 22, NULL, 241, 1, 2, 91, 1, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1660, N'King Solomon''s Mines', CAST(N'1985-11-22T00:00:00.000' AS DateTime), 321, 264, NULL, 241, 1, 8, 100, 2, 11000000, 15057000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1661, N'King Solomon''s Mines', CAST(N'1937-06-17T00:00:00.000' AS DateTime), 293, 18, NULL, 240, 1, 8, 80, 1, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1662, N'East of Eden', CAST(N'1955-03-09T00:00:00.000' AS DateTime), 341, 6, NULL, 241, 1, 2, 117, 2, NULL, 5000000, 4, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1663, N'Giant', CAST(N'1956-11-24T00:00:00.000' AS DateTime), 338, 6, NULL, 241, 1, 1, 201, 2, 5400000, 39000000, 10, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1664, N'Rebel Without a Cause', CAST(N'1955-10-27T00:00:00.000' AS DateTime), 507, 6, NULL, 241, 1, 2, 111, 2, 1500000, 4500000, 3, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1665, N'Miracle on 34th Street', CAST(N'1994-11-18T00:00:00.000' AS DateTime), 508, 148, NULL, 241, 1, 26, 114, 1, NULL, 46300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1666, N'Gentleman''s Agreement', CAST(N'1947-11-11T00:00:00.000' AS DateTime), 341, 4, NULL, 241, 1, 2, 118, 2, 1985000, 7800000, 8, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1667, N'Viva Zapata!', CAST(N'1952-02-07T00:00:00.000' AS DateTime), 341, 4, NULL, 241, 1, 19, 113, 2, 1800000, 1900000, 5, 1);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1668, N'On the Waterfront', CAST(N'1954-07-28T00:00:00.000' AS DateTime), 341, 170, NULL, 241, 1, 14, 108, 2, 910000, 9600000, 12, 8);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1669, N'Ice Cold in Alex', CAST(N'1958-06-24T00:00:00.000' AS DateTime), 321, 172, NULL, 240, 1, 5, 130, 2, NULL, NULL, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1670, N'The Diary of Anne Frank', CAST(N'1959-03-18T00:00:00.000' AS DateTime), 338, 4, NULL, 241, 1, 2, 180, 1, 3800000, 2300000, 8, 3);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1671, N'The Greatest Story Ever Told', CAST(N'1965-04-09T00:00:00.000' AS DateTime), 338, 22, N'Well that''s wrong - everyone knows that the greatest story ever told is Predator.', 241, 1, 2, 260, 1, 21000000, 15473000, 5, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1672, N'A Place in the Sun', CAST(N'1951-08-14T00:00:00.000' AS DateTime), 338, 2, NULL, 241, 1, 2, 122, 2, 2295000, 7000000, 9, 6);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1673, N'The Talk of the Town', CAST(N'1942-08-20T00:00:00.000' AS DateTime), 338, 5, NULL, 241, 1, 7, 118, 1, 1000000, 1100000, 7, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1674, N'Samson and Delilah', CAST(N'1949-12-21T00:00:00.000' AS DateTime), 127, 2, NULL, 241, 1, 2, 128, 1, 2900000, 25600000, 5, 2);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1675, N'The NeverEnding Story', CAST(N'1984-04-06T00:00:00.000' AS DateTime), 25, 35, N'Don''t be fooled by the title: it ends after about an hour and half.', 257, 1, 12, 93, 1, 27000000, 100000000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1676, N'Enemy Mine', CAST(N'1985-12-20T00:00:00.000' AS DateTime), 25, 238, NULL, 241, 1, 3, 108, 5, 29000000, 12300000, 0, 0);
INSERT Film (FilmID, Title, ReleaseDate, DirectorID, StudioID, Review, CountryID, LanguageID, GenreID, RunTimeMinutes, CertificateID, BudgetDollars, BoxOfficeDollars, OscarNominations, OscarWins) VALUES (1677, N'In the Line of Fire', CAST(N'1993-07-09T00:00:00.000' AS DateTime), 25, 149, NULL, 241, 1, 9, 128, 5, 40000000, 177000000, 3, 0);


-- Create database if not exists employees ;

-- use employees ;
drop table if exists employees;
CREATE TABLE Employees(
	EmployeeID int not NULL PRIMARY KEY,
	First nvarchar(50) NULL,
	Last nvarchar(50) NULL,
	Building nvarchar(50) NULL,
	Department nvarchar(50) NULL,
	Status nvarchar(50) NULL,
	HireDate datetime NULL,
	BenefitDental nvarchar(50) NULL,
	BenefitMedical nvarchar(50) NULL,
	BenefitRetirement nvarchar(50) NULL,
	Salary int NULL,
	JobRating int NULL
);

Insert into Employees (EmployeeID,First,Last,Building,Department,Status,HireDate,BenefitDental,BenefitMedical,BenefitRetirement,Salary,JobRating) Values
(1,'Zachary','Tate','Taft','Manufacturing','Full Time','1999-06-10','Yes','Yes','Yes',78570,1),
(2,'Robert','Wilcox','Main','Quality Assurance','Contract','1999-06-15','No','No','No',86100,4),
(3,'Teresa','Baxter','Taft','Manufacturing','Half-Time','1999-06-16','Yes','Yes','Yes',48835,5),
(4,'Suzanne','Farmer','West','Account Management','Full Time','1999-06-30','No','Yes','No',22870,3),
(5,'Leonard','Grant','South','Account Management','Half-Time','1999-07-05','Yes','Yes','Yes',31205,2),
(6,'David','Armstrong','North','Quality Control','Contract','1999-07-05','No','No','No',45030,3),
(7,'Tia','Powers','North','Facilities/Engineering','Full Time','1999-07-08','Yes','Yes','Yes',61030,3),
(8,'David','Meyers','West','Manufacturing','Hourly','1999-07-08','No','No','No',29176,3),
(9,'Mike','Walker','Watson','Manufacturing','Full Time','1999-07-09','No','Yes','No',35460,5),
(10,'Marc','Navarro','Main','Facilities/Engineering','Half-Time','1999-07-21','Yes','No','No',23380,4),
(11,'Lawrence','Pugh','South','Quality Assurance','Half-Time','1999-07-21','No','Yes','No',39530,5),
(12,'Bryan','Briggs','Main','Facilities/Engineering','Contract','1999-07-27','No','No','No',85480,5),
(13,'Corey','Patton','West','Marketing','Full Time','1999-08-04','Yes','No','No',86830,3),
(14,'Thomas','Chavez','Main','Marketing','Full Time','1999-08-05','No','No','Yes',82110,3),
(15,'Grant','Barnes','Main','Quality Assurance','Hourly','1999-08-09','No','No','No',12836,5),
(16,'Ray','Turner','West','Quality Assurance','Hourly','1999-08-17','No','No','No',17912,5),
(17,'Craig','Henry','North','Quality Assurance','Full Time','1999-09-13','No','No','Yes',70280,3),
(18,'Ronald','Lawrence','Main','Facilities/Engineering','Contract','1999-09-19','No','No','No',63340,3),
(19,'Chris','Garrison','North','IT','Full Time','1999-09-20','Yes','Yes','Yes',68520,5),
(20,'Elaine','Jefferson','Taft','Account Management','Full Time','1999-10-05','Yes','Yes','No',73740,4),
(21,'James','Dudley','South','IT','Contract','1999-10-08','No','No','No',63330,4),
(22,'Cathy','Fields','Taft','Manufacturing','Contract','1999-10-14','No','No','No',76910,1),
(23,'Cherie','Burgess','North','Manufacturing','Full Time','1999-10-19','No','No','Yes',89740,5),
(24,'Scott','Melton','West','Manufacturing','Full Time','1999-10-25','No','Yes','No',55450,5),
(25,'Rick','Gallegos','Main','Product Development','Contract','1999-10-26','No','No','No',25120,5),
(26,'Timothy','Wallace','Watson','Facilities/Engineering','Contract','1999-10-27','No','No','No',88000,5),
(27,'Raquel','Booth','West','Account Management','Hourly','1999-11-02','No','No','No',14332,5),
(28,'Eric','Jackson','North','Account Management','Full Time','1999-11-14','No','No','Yes',61420,4),
(29,'Eileen','Roberson','Main','Creative','Contract','1999-11-15','No','No','No',64470,3),
(30,'Robert','Owen','Main','Sales','Half-Time','1999-11-17','Yes','Yes','Yes',20500,3),
(31,'Erin','Harding','West','Facilities/Engineering','Full Time','1999-11-21','No','Yes','No',34780,4),
(32,'Mark','Russell','Main','IT','Full Time','1999-11-25','Yes','Yes','Yes',74530,5),
(33,'Raymond','Bradford','North','Manufacturing','Contract','1999-11-28','No','No','No',71710,5),
(34,'Kevin','Atkins','North','Manufacturing','Hourly','1999-12-08','No','No','No',8904,3),
(35,'Garret','Santos','Taft','Product Development','Full Time','1999-12-15','Yes','Yes','No',71010,5),
(36,'Chris','Preston','Main','Account Management','Full Time','1999-12-16','Yes','No','No',36890,1),
(37,'Richard','Chambers','Main','Manufacturing','Contract','1999-12-18','No','No','No',68260,5),
(38,'Florianne','Jacobs','Main','Manufacturing','Full Time','1999-12-22','Yes','Yes','Yes',40340,2),
(39,'Gregory','Wilkinson','North','Manufacturing','Contract','1999-12-23','No','No','No',72520,3),
(40,'Rhonda','Stafford','Main','Manufacturing','Full Time','2000-01-04','Yes','Yes','Yes',75120,5),
(41,'Carl','Parker','South','Manufacturing','Contract','2000-01-06','No','No','No',45050,1),
(42,'Jen','Middleton','Main','Quality Assurance','Full Time','2000-01-07','Yes','Yes','No',78950,1),
(43,'Ronald','Crawford','North','Quality Control','Full Time','2000-01-20','Yes','Yes','No',64470,5),
(44,'Diane','Andrews','North','Account Management','Hourly','2000-01-23','No','No','No',16688,3),
(45,'Brett','Conway','South','Facilities/Engineering','Contract','2000-02-02','No','No','No',30340,3),
(46,'Ryan','McCarthy','West','Sales','Contract','2000-02-11','No','No','No',72480,2),
(47,'Janene','Cruz','South','Quality Control','Half-Time','2000-02-17','Yes','No','No',46105,5),
(48,'Bobby','Mendoza','West','Facilities/Engineering','Hourly','2000-02-23','No','No','No',18500,5),
(49,'Erik','Pena','South','Manufacturing','Hourly','2000-03-02','No','No','No',37612,4),
(50,'Hsi','McCormick','West','Quality Assurance','Contract','2000-03-05','No','No','No',59050,4),
(51,'Scott','McDowell','Main','Creative','Full Time','2000-03-12','No','No','Yes',50110,1),
(52,'Kenneth','Miles','Main','Quality Assurance','Full Time','2000-03-13','Yes','No','No',79610,2),
(53,'Shannon','Thomas','Taft','Quality Control','Full Time','2000-03-15','No','No','Yes',48410,5),
(54,'Mark','Massey','West','Quality Assurance','Full Time','2000-03-16','No','No','Yes',67407,5),
(55,'Jaime','Chen','Taft','Product Development','Full Time','2000-03-17','No','No','Yes',82400,2),
(56,'Amy','Decker','South','Manufacturing','Full Time','2000-03-19','Yes','Yes','Yes',33210,4),
(57,'Duane','Pope','Main','Sales','Hourly','2000-03-19','No','No','No',21668,4),
(58,'Jessica','Chapman','North','Account Management','Half-Time','2000-03-19','Yes','Yes','No',45750,5),
(59,'Michael','Bowman','West','Account Management','Contract','2000-03-21','No','No','No',47520,1),
(60,'Christopher','Perry','West','Facilities/Engineering','Contract','2000-03-22','No','No','No',76930,1),
(61,'Mark','Moses','Main','Account Management','Full Time','2000-03-23','No','Yes','No',54580,4),
(62,'Alfred','Waters','North','Marketing','Half-Time','2000-03-29','Yes','Yes','Yes',47885,1),
(63,'Debra','Parrish','Main','Manufacturing','Full Time','2000-04-05','Yes','Yes','Yes',88240,5),
(64,'Jennifer','Barr','North','Marketing','Full Time','2000-04-20','Yes','No','No',49770,1),
(65,'Kristina','Jensen','West','Marketing','Half-Time','2000-04-25','No','No','Yes',28880,3),
(66,'Anthony','Carson','West','Manufacturing','Full Time','2000-04-26','Yes','Yes','No',22660,2),
(67,'Leslie','Doyle','Taft','Account Management','Full Time','2000-05-10','No','No','Yes',45000,4),
(68,'Debbie','Wolf','Taft','Training','Full Time','2000-05-17','No','No','Yes',72090,5),
(69,'Robert','Moore','West','Marketing','Full Time','2000-05-19','Yes','Yes','No',31260,5),
(70,'Lisa','Page','West','ADC','Full Time','2000-06-14','Yes','Yes','Yes',24550,1),
(71,'Emily','Leon','Main','Sales','Full Time','2000-06-18','No','No','Yes',23520,2),
(72,'Jill','Maxwell','Taft','Creative','Contract','2000-06-19','No','No','No',32940,5),
(73,'David','Mercado','Main','Manufacturing','Half-Time','2000-06-20','No','Yes','No',21670,2),
(74,'Ed','Gomez','South','Environmental Health/Safety','Full Time','2000-06-25','No','No','Yes',68300,5),
(75,'Thomas','Marquez','West','Marketing','Contract','2000-07-05','No','No','No',47620,5),
(76,'Brandon','Blackwell','West','Quality Control','Contract','2000-07-06','No','No','No',58250,2),
(77,'Allan','Carter','Main','Account Management','Full Time','2000-07-08','No','Yes','No',46360,5),
(78,'Tamara','McDaniel','Main','Quality Assurance','Half-Time','2000-07-09','Yes','Yes','Yes',34980,2),
(79,'Roy','Butler','Taft','Manufacturing','Full Time','2000-07-11','Yes','No','No',81400,2),
(80,'Mary','Reyes','Taft','Marketing','Contract','2000-07-12','No','No','No',31270,5),
(81,'Matthew','Dunn','North','Environmental Compliance','Full Time','2000-07-27','Yes','No','No',47850,1),
(82,'Brian D','Hoffman','North','Quality Assurance','Contract','2000-07-27','No','No','No',53310,5),
(83,'Kent','Oconnor','Taft','Training','Half-Time','2000-07-30','Yes','Yes','Yes',15240,1),
(84,'Michael','Kirby','South','Account Management','Half-Time','2000-07-30','No','No','Yes',22475,4),
(85,'Terry','Palmer','South','Quality Control','Full Time','2000-08-25','Yes','No','No',77680,3),
(86,'Michael','Solomon','North','Quality Assurance','Full Time','2000-08-27','Yes','Yes','Yes',58410,5),
(87,'Peter','Foley','Main','Environmental Health/Safety','Full Time','2000-08-31','Yes','Yes','Yes',49860,2),
(88,'Frank','Wiggins','North','Training','Full Time','2000-09-11','Yes','Yes','Yes',75150,1),
(89,'Chris','Montgomery','South','Product Development','Hourly','2000-09-14','No','No','No',38768,4),
(90,'Shaun','Winters','North','Creative','Full Time','2000-09-20','No','No','Yes',61330,1),
(91,'Todd','Francis','Main','Manufacturing','Half-Time','2000-09-20','No','No','Yes',48190,1),
(92,'Brad','Wright','Main','Manufacturing','Full Time','2000-09-24','No','No','Yes',61330,4),
(93,'Bacardi','Pittman','South','Manufacturing','Contract','2000-10-04','No','No','No',25130,5),
(94,'Kevin','Aguilar','West','Quality Assurance','Full Time','2000-10-11','No','No','Yes',39000,3),
(95,'Danny','Love','Watson','Marketing','Full Time','2000-10-18','Yes','Yes','Yes',46030,2),
(96,'Christopher','Soto','Main','Account Management','Contract','2000-10-18','No','No','No',57990,5),
(97,'Thomas','Hurst','Main','Manufacturing','Hourly','2000-10-26','No','No','No',9424,4),
(98,'Sheela','Gaines','North','Quality Assurance','Full Time','2000-11-02','Yes','Yes','Yes',69200,4),
(99,'Robert','Sullivan','North','Manufacturing','Full Time','2000-11-08','No','No','Yes',68750,1),
(100,'Diane','Chandler','Main','IT','Hourly','2000-11-19','No','No','No',39764,1),
(101,'Curtis','Sheppard','Main','Quality Control','Full Time','2000-11-20','Yes','No','No',61850,2),
(102,'Robbie','Barber','North','Manufacturing','Full Time','2000-11-22','No','Yes','No',37760,2),
(103,'Deborah','Lindsey','North','Account Management','Hourly','2000-11-30','No','No','No',37016,4),
(104,'Roger','Copeland','Watson','Creative','Contract','2000-12-02','No','No','No',86970,4),
(105,'Cassy','Pearson','Watson','Manufacturing','Contract','2000-12-10','No','No','No',27380,3),
(106,'Stephen','Silva','Main','Creative','Half-Time','2000-12-17','Yes','No','No',42905,1),
(107,'Carol','Moran','North','Manufacturing','Half-Time','2000-12-20','No','No','Yes',11065,1),
(108,'Randall','Ruiz','West','Quality Assurance','Full Time','2000-12-20','No','No','Yes',67020,1),
(109,'Christopher','Parks','Taft','Quality Control','Half-Time','2000-12-25','No','Yes','No',19825,2),
(110,'Janice','Ross','South','Marketing','Half-Time','2000-12-31','No','No','Yes',26790,2),
(111,'Teason','Anderson','Watson','Account Management','Contract','2001-01-04','No','No','No',71300,5),
(112,'Douglas','Howell','Main','Account Management','Hourly','2001-01-09','No','No','No',36052,5),
(113,'John','Jones','Taft','Marketing','Full Time','2001-01-21','No','No','Yes',51410,4),
(114,'Bryant','Vargas','Main','Account Management','Full Time','2001-01-22','No','No','Yes',65910,5),
(115,'Paul','Dennis','Main','Quality Assurance','Contract','2001-02-04','No','No','No',60800,4),
(116,'Dean','Collier','North','Quality Assurance','Full Time','2001-02-05','Yes','Yes','No',45880,5),
(117,'James','Finley','South','Manufacturing','Full Time','2001-02-11','Yes','Yes','Yes',40060,3),
(118,'Shari','Mathis','South','Product Development','Full Time','2001-02-12','Yes','No','No',39680,5),
(119,'Vicky','Payne','North','Manufacturing','Half-Time','2001-02-20','No','No','Yes',17270,5),
(120,'Michael','Santiago','North','Quality Assurance','Half-Time','2001-02-21','Yes','No','No',47350,1),
(121,'Chris','McKinney','North','Manufacturing','Full Time','2001-03-01','Yes','Yes','Yes',35820,2),
(122,'Edward','McLaughlin','West','Manufacturing','Full Time','2001-03-14','Yes','Yes','Yes',67280,3),
(123,'David','Diaz','Main','Manufacturing','Contract','2001-03-25','No','No','No',23810,4),
(124,'Pete','Caldwell','Main','Marketing','Full Time','2001-03-26','No','No','Yes',43460,5),
(125,'Lara','Larsen','Watson','Quality Control','Hourly','2001-03-28','No','No','No',32536,2),
(126,'April','Goodwin','North','Quality Control','Half-Time','2001-04-01','No','No','Yes',26185,5),
(127,'Douglas','Bryant','North','Marketing','Full Time','2001-04-02','Yes','Yes','Yes',44220,3),
(128,'Alicia','Franklin','Watson','Sales','Full Time','2001-04-05','No','No','Yes',68410,5),
(129,'Lorna','Hudson','West','Facilities/Engineering','Contract','2001-04-09','No','No','No',23560,3),
(130,'Gayla','Dean','North','Quality Assurance','Contract','2001-04-18','No','No','No',54840,4),
(131,'Steve','May','West','Facilities/Engineering','Hourly','2001-04-26','No','No','No',33056,5),
(132,'Eugene','Glover','Main','Manufacturing','Half-Time','2001-05-12','No','Yes','No',41615,1),
(133,'Gregory','Clayton','North','Facilities/Engineering','Full Time','2001-05-15','No','No','Yes',32100,1),
(134,'Ray','Baldwin','Watson','Quality Assurance','Full Time','2001-05-23','No','No','Yes',48250,3),
(135,'Richard','Campos','Main','Major Mfg Projects','Hourly','2001-05-28','No','No','Yes',61860,5),
(136,'James','Abbott','North','Quality Assurance','Full Time','2001-06-04','Yes','Yes','Yes',29260,4),
(137,'Chris','Kirk','Main','Quality Assurance','Half-Time','2001-06-09','No','Yes','No',20990,4),
(138,'Donald','Patel','Main','Manufacturing','Full Time','2001-06-13','Yes','Yes','Yes',76192,4),
(139,'Edward','Hayes','Main','Account Management','Full Time','2001-06-14','Yes','Yes','Yes',62400,4),
(140,'John','Barrett','North','Quality Control','Full Time','2001-06-27','No','No','Yes',35460,1),
(141,'Ronald','Carpenter','Watson','Quality Assurance','Hourly','2001-07-05','No','No','No',15552,4),
(142,'Kim','Green','Watson','Research/Development','Half-Time','2001-07-05','Yes','Yes','Yes',31250,2),
(143,'Icelita','Kelly','South','Sales','Full Time','2001-07-15','Yes','No','No',45450,5),
(144,'Shira','Nelson','West','Human Resources','Full Time','2001-07-17','No','No','Yes',75060,5),
(145,'Elbert','Cole','South','Manufacturing','Contract','2001-08-17','No','No','No',41840,2),
(146,'Diane','Matthews','North','Facilities/Engineering','Hourly','2001-08-19','No','No','No',30080,3),
(147,'Debra','Boyd','North','Manufacturing','Half-Time','2001-08-21','No','No','Yes',46710,3),
(148,'Lesa','Carroll','West','Environmental Compliance','Full Time','2001-09-05','No','Yes','No',56440,1),
(149,'Michael','Welch','North','Quality Control','Full Time','2001-09-05','Yes','Yes','No',71970,4),
(150,'Joseph','Pace','Main','IT','Contract','2001-09-09','No','No','No',30300,1),
(151,'John','Sexton','Main','Manufacturing','Contract','2001-09-23','No','No','No',57600,3),
(152,'Don','Guzman','Main','Quality Assurance','Contract','2001-09-28','No','No','No',77760,3),
(153,'Charles','Lee','West','Product Development','Full Time','2001-09-29','No','No','Yes',71380,2),
(154,'Michael','Campbell','West','Marketing','Full Time','2001-10-18','No','No','Yes',52490,4),
(155,'Timothy','Beasley','North','Quality Control','Full Time','2001-10-28','Yes','Yes','No',65320,5),
(156,'Kristin','Randolph','South','Major Mfg Projects','Contract','2001-10-29','No','No','Yes',69410,4),
(157,'Al','Serrano','Main','Account Management','Full Time','2001-10-29','Yes','No','No',48330,1),
(158,'Michael','Mosley','North','IT','Half-Time','2001-11-20','No','No','Yes',29005,1),
(159,'Andrew','Humphrey','West','Manufacturing','Full Time','2001-11-23','No','Yes','No',23650,1),
(160,'Richard','McLean','North','Marketing','Contract','2001-11-28','No','No','No',50200,4),
(161,'Kathleen','Browning','South','Manufacturing','Contract','2001-11-29','No','No','No',57760,3),
(162,'Jessica','Whitaker','North','Manufacturing','Full Time','2001-12-02','Yes','No','No',38870,2),
(163,'Ignacio','Huffman','North','Manufacturing','Contract','2001-12-13','No','No','No',89520,5),
(164,'Anita','Marshall','North','Manufacturing','Contract','2001-12-24','No','No','No',45420,1),
(165,'Daniel','White','Watson','Training','Full Time','2002-01-28','Yes','No','No',74840,4),
(166,'Charles','Alexander','West','Manufacturing Admin','Contract','2002-01-29','No','No','No',74500,4),
(167,'Ken','Rowe','West','Training','Contract','2002-02-10','No','No','No',76690,3),
(168,'John','Cameron','North','Account Management','Contract','2002-02-20','No','No','No',89640,4),
(169,'David','Gonzalez','Taft','Quality Control','Full Time','2002-04-17','No','No','Yes',47630,3),
(170,'Charles','Thornton','West','Environmental Compliance','Full Time','2002-06-06','No','No','Yes',33640,3),
(171,'Jimmy','Simpson','Watson','Facilities/Engineering','Half-Time','2002-06-09','Yes','Yes','Yes',35280,3),
(172,'Harlon','Vaughn','Main','Sales','Full Time','2002-06-11','Yes','Yes','Yes',71820,2),
(173,'Cinnamon','Quinn','North','Quality Control','Half-Time','2002-07-01','Yes','Yes','Yes',17205,5),
(174,'Barbara','Reynolds','West','Marketing','Full Time','2002-07-23','Yes','Yes','Yes',48990,5),
(175,'Janet','Daniels','North','Quality Control','Full Time','2002-08-08','Yes','No','No',49930,1),
(176,'Michael','Graves','North','Quality Control','Full Time','2002-08-19','Yes','Yes','Yes',63060,4),
(177,'David','Larson','North','Manufacturing','Contract','2002-08-29','No','No','No',68510,5),
(178,'Lisa','Montoya','Main','Research Center','Full Time','2002-09-12','Yes','Yes','Yes',63670,5),
(179,'Karin','Sherman','North','Facilities/Engineering','Full Time','2002-09-29','Yes','Yes','Yes',27180,4),
(180,'Gary','Pennington','Taft','Quality Control','Full Time','2002-09-30','No','No','Yes',78710,2),
(181,'Nathaniel','Nichols','North','Facilities/Engineering','Full Time','2002-10-09','No','No','Yes',28650,4),
(182,'Elena','Miranda','South','Major Mfg Projects','Full Time','2002-11-03','Yes','Yes','No',45150,1),
(183,'Thomas','Keith','Watson','Quality Control','Contract','2002-11-25','No','No','No',77136,5),
(184,'Verna','Bates','North','Manufacturing','Full Time','2002-11-28','Yes','Yes','No',66010,5),
(185,'Jenny','Leblanc','West','Research Center','Full Time','2002-12-03','Yes','No','No',40680,5),
(186,'Stephen','Warner','Taft','Quality Assurance','Contract','2002-12-12','No','No','No',46780,2),
(187,'Kimberly','Kennedy','Main','Quality Control','Contract','2002-12-29','No','No','No',28270,5),
(188,'Michael','Molina','Main','Creative','Full Time','2003-01-12','No','Yes','No',61150,4),
(189,'Daniel','Johnston','West','Marketing','Half-Time','2003-02-06','Yes','Yes','No',31110,1),
(190,'Jessica','Wilson','Main','Account Management','Full Time','2003-02-06','Yes','Yes','Yes',29130,1),
(191,'James','Tucker','South','Marketing','Half-Time','2003-02-09','Yes','No','No',15910,3),
(192,'Dominic','Jimenez','Watson','Quality Control','Contract','2003-02-09','No','No','No',25530,3),
(193,'Troy','Weeks','Main','Marketing','Half-Time','2003-03-06','Yes','Yes','No',47295,4),
(194,'Brian','Fletcher','Main','Facilities/Engineering','Full Time','2003-03-16','No','Yes','No',62790,2),
(195,'Charles','Meyer','South','Manufacturing','Full Time','2003-05-08','No','No','Yes',25310,4),
(196,'Matthew','Moody','West','Sales','Contract','2003-05-15','No','No','No',29540,3),
(197,'Kathleen','Martinez','Main','Facilities/Engineering','Full Time','2003-05-20','Yes','Yes','Yes',71950,5),
(198,'Hunyen','Curry','Main','Account Management','Half-Time','2003-05-28','Yes','No','No',12545,4),
(199,'Benjamin','Young','Main','Product Development','Full Time','2003-07-06','Yes','Yes','Yes',42480,3),
(200,'Tony','Roth','Taft','Manufacturing','Contract','2003-08-13','No','No','No',52770,2),
(201,'Elizabeth','Stewart','Main','Manufacturing','Full Time','2003-08-18','No','No','Yes',62750,3),
(202,'Paul','Harmon','North','Facilities/Engineering','Full Time','2003-09-04','Yes','Yes','No',85880,3),
(203,'Donnie','Colon','North','Manufacturing','Full Time','2003-10-20','Yes','Yes','Yes',28970,3),
(204,'Timothy','Tanner','Taft','Training','Full Time','2003-10-30','Yes','Yes','Yes',30780,4),
(205,'Michael','Vasquez','Main','Account Management','Contract','2003-10-30','No','No','No',60070,3),
(206,'Larry','Wood','Taft','Manufacturing Admin','Full Time','2003-11-02','Yes','Yes','Yes',59140,5),
(207,'Steve','Little','Main','Manufacturing','Full Time','2003-12-01','No','Yes','No',64130,1),
(208,'Ruben','Salazar','Watson','Quality Control','Contract','2003-12-18','No','No','No',49090,4),
(209,'Chris','Bauer','West','IT','Half-Time','2004-01-04','Yes','Yes','Yes',33810,5),
(210,'Philip','Mueller','Taft','Quality Control','Full Time','2004-01-29','Yes','No','No',60300,2),
(211,'Terry','Gilmore','North','Major Mfg Projects','Half-Time','2004-02-08','Yes','No','No',51800,1),
(212,'Desiree','Snow','Main','Manufacturing','Full Time','2004-02-12','No','No','Yes',69080,3),
(213,'Brian','Morton','North','Account Management','Contract','2004-02-29','No','No','No',61580,3),
(214,'Gary','Jennings','North','Account Management','Full Time','2004-04-11','Yes','Yes','No',45100,2),
(215,'Robin','Hutchinson','Watson','Creative','Full Time','2004-05-25','No','Yes','No',39740,1),
(216,'Ernest','Haynes','West','Manufacturing','Half-Time','2004-06-02','Yes','Yes','Yes',24460,1),
(217,'Cliff','Black','Watson','Quality Assurance','Full Time','2004-06-07','No','No','Yes',82490,5),
(218,'Scott','Rodriguez','North','Manufacturing','Contract','2004-06-16','No','No','No',61370,3),
(219,'Justin','Bass','North','Product Development','Contract','2004-06-23','No','No','No',31970,5),
(220,'Troy','Durham','South','Creative','Contract','2004-07-19','No','No','No',73390,2),
(221,'Victor','Bailey','Watson','Research Center','Full Time','2004-08-05','No','No','Yes',42800,5),
(222,'Danielle','Atkinson','North','Account Management','Full Time','2004-08-22','Yes','Yes','No',23560,3),
(223,'Ryan','Banks','Main','IT','Hourly','2004-09-01','No','No','No',21648,2),
(224,'Lincoln','Dickerson','North','Manufacturing','Hourly','2004-09-20','No','No','No',8892,1),
(225,'Lateef','Rush','West','Quality Assurance','Half-Time','2004-11-04','Yes','Yes','No',28525,4),
(226,'Rajean','Strickland','Main','Training','Half-Time','2004-11-11','Yes','No','No',17735,3),
(227,'Renee','Hood','North','Facilities/Engineering','Full Time','2004-11-14','No','No','Yes',87280,4),
(228,'Michael','Munoz','South','Quality Assurance','Full Time','2004-11-22','Yes','Yes','Yes',29210,5),
(229,'John','Lucas','North','Sales','Contract','2004-12-02','No','No','No',78100,3),
(230,'Troy','Davenport','North','Quality Control','Full Time','2004-12-09','No','No','Yes',48010,3),
(231,'Christopher','Glenn','South','Quality Assurance','Half-Time','2004-12-14','Yes','Yes','Yes',48740,1),
(232,'William','Clay','Watson','Manufacturing','Contract','2004-12-19','No','No','No',75420,1),
(233,'Asa','Byrd','Watson','Sales','Hourly','2004-12-26','No','No','No',11044,2),
(234,'Gary','Long','South','Account Management','Full Time','2005-01-16','Yes','Yes','No',76910,2),
(235,'Eric','Weaver','North','Manufacturing','Full Time','2005-02-03','Yes','Yes','No',54230,5),
(236,'Neill','Juarez','North','Green Building','Full Time','2005-02-20','Yes','Yes','Yes',86530,1),
(237,'Jaime','Davidson','Watson','Manufacturing','Contract','2005-03-08','No','No','No',64220,5),
(238,'Deborah','Heath','Watson','Manufacturing','Full Time','2005-04-14','No','No','Yes',30920,5),
(239,'Glenn','Hernandez','North','Manufacturing Admin','Full Time','2005-04-14','No','No','Yes',53870,2),
(240,'David','Day','Main','Manufacturing','Full Time','2005-04-21','Yes','Yes','Yes',75176,3),
(241,'Kenneth','Douglas','North','Major Mfg Projects','Hourly','2005-04-24','Yes','Yes','Yes',85130,5),
(242,'Gretchen','Becker','West','Facilities/Engineering','Full Time','2005-05-08','Yes','Yes','Yes',66890,5),
(243,'Andrew','Stevens','Taft','Quality Assurance','Contract','2005-07-14','No','No','No',64590,1),
(244,'Robert','Carrillo','North','Manufacturing','Contract','2005-07-31','No','No','No',57410,2),
(245,'Dawn','Potter','Taft','Environmental Compliance','Full Time','2005-08-07','Yes','Yes','Yes',30350,1),
(246,'John','Robinson','North','Manufacturing','Contract','2005-08-29','No','No','No',39300,2),
(247,'Michael','Morse','North','Product Development','Full Time','2005-10-30','No','Yes','No',65560,1),
(248,'Troy','Chase','Main','Training','Full Time','2005-11-06','Yes','Yes','Yes',49350,4),
(249,'Suzanne','Robbins','North','IT','Hourly','2005-11-08','No','No','No',33512,4),
(250,'Clint','Velasquez','West','Manufacturing','Full Time','2005-11-10','Yes','Yes','Yes',47340,2),
(251,'Michael','Schmidt','North','Account Management','Half-Time','2005-12-07','No','No','Yes',32900,2),
(252,'Lara','Best','Main','IT','Full Time','2006-01-30','No','No','Yes',86200,3),
(253,'Michael','Paul','West','Quality Assurance','Full Time','2006-02-09','No','No','Yes',31910,5),
(254,'John','Manning','West','Manufacturing','Contract','2006-04-02','No','No','No',71830,3),
(255,'Duane','Dominguez','North','Manufacturing','Contract','2006-05-04','No','No','No',37980,4),
(256,'Angela','Flores','Watson','Manufacturing','Full Time','2006-05-04','Yes','Yes','No',70760,1),
(257,'Gary','McClure','North','Quality Control','Full Time','2006-05-11','Yes','Yes','No',48280,4),
(258,'Frederick','Lewis','West','Quality Control','Full Time','2006-05-30','No','No','Yes',81340,2),
(259,'George','Horn','Watson','Account Management','Half-Time','2007-06-10','No','No','Yes',10630,3),
(260,'Richard','Dixon','West','Manufacturing','Full Time','2007-06-20','Yes','No','No',68710,4),
(261,'Rita','Norman','North','Sales','Contract','2007-06-21','No','No','No',54190,4),
(262,'Bruce','Norton','Main','Environmental Health/Safety','Full Time','2007-06-23','No','No','Yes',22920,3),
(263,'Karen','Logan','Main','Manufacturing','Contract','2007-06-25','No','No','No',42150,5),
(264,'Denise','Knight','West','Human Resources','Contract','2007-06-25','No','No','No',25120,2),
(265,'Colleen','Rogers','West','Quality Assurance','Full Time','2007-06-25','Yes','No','No',62965,1),
(266,'Michelle','McKenzie','Main','Environmental Compliance','Full Time','2007-07-03','No','No','Yes',49360,2),
(267,'Erik','Pratt','North','Training','Full Time','2007-07-08','Yes','Yes','Yes',60830,2),
(268,'Benning','Nunez','North','Product Development','Full Time','2007-07-10','Yes','Yes','Yes',22410,4),
(269,'William','Hensley','North','Quality Control','Half-Time','2007-07-10','Yes','Yes','No',37660,4),
(270,'Gregory','Hardin','North','Green Building','Contract','2007-07-12','No','No','No',78860,2),
(271,'Terry','Garner','Watson','Creative','Full Time','2007-07-31','Yes','Yes','Yes',80120,4),
(272,'Janine','Sharp','West','Manufacturing','Hourly','2007-08-03','No','No','No',22472,1),
(273,'Roger','Blankenship','North','Quality Assurance','Full Time','2007-08-10','Yes','Yes','Yes',78710,4),
(274,'Nobuko','Shaffer','North','Product Development','Full Time','2007-08-14','No','No','Yes',37750,5),
(275,'Shannon','Gilbert','Watson','Account Management','Full Time','2007-08-16','Yes','Yes','No',62688,3),
(276,'Randy','Pruitt','Main','Facilities/Engineering','Contract','2007-08-18','No','No','No',74740,5),
(277,'William','Clark','North','Quality Assurance','Contract','2007-08-19','No','No','No',85930,2),
(278,'John','Boyer','West','Manufacturing','Full Time','2007-08-24','No','No','Yes',73144,5),
(279,'Deborah','Kim','Watson','Environmental Health/Safety','Full Time','2007-08-27','No','Yes','No',26510,1),
(280,'Lee','Nicholson','West','Facilities/Engineering','Half-Time','2007-08-30','No','Yes','No',48415,4),
(281,'Bonnie','Stephens','Watson','Marketing','Contract','2007-08-31','No','No','No',53870,2),
(282,'Faye','Kramer','West','Quality Control','Half-Time','2007-08-31','Yes','Yes','No',13690,5),
(283,'Thomas','Allen','Main','Manufacturing','Full Time','2007-09-02','Yes','Yes','Yes',79730,2),
(284,'John','Fowler','North','IT','Full Time','2007-09-02','Yes','Yes','Yes',47060,4),
(285,'Brian','Harris','West','Manufacturing','Full Time','2007-09-04','Yes','Yes','No',76584,1),
(286,'Marie','Fernandez','North','Manufacturing','Full Time','2007-09-08','No','No','Yes',32390,2),
(287,'Dennis','Freeman','North','Account Management','Full Time','2007-09-10','Yes','Yes','Yes',63270,1),
(288,'Mark','Houston','South','Manufacturing','Full Time','2007-09-11','No','Yes','No',44920,1),
(289,'Carol','McGee','North','Manufacturing','Full Time','2007-09-16','No','No','Yes',65720,1),
(290,'Robert','French','West','Account Management','Contract','2007-09-23','No','No','No',49530,4),
(291,'Scott','Hobbs','North','Manufacturing','Full Time','2007-09-27','Yes','No','No',29420,5),
(292,'Larry','Reed','North','Marketing','Full Time','2007-09-29','Yes','Yes','Yes',81640,4),
(293,'Irv','Fleming','West','Environmental Compliance','Half-Time','2007-10-16','Yes','Yes','Yes',11025,1),
(294,'John','Wall','Main','Human Resources','Contract','2007-10-19','No','No','No',44820,4),
(295,'Angus','Kent','North','Facilities/Engineering','Contract','2007-10-21','No','No','No',84200,2),
(296,'LaReina','Marks','North','Quality Assurance','Full Time','2007-10-21','No','No','Yes',37770,5),
(297,'James','Sandoval','North','Quality Control','Hourly','2007-10-28','No','No','No',28768,3),
(298,'Paul','House','South','Manufacturing','Contract','2007-11-08','No','No','No',59330,4),
(299,'Bill','Osborne','Main','Manufacturing','Full Time','2007-11-10','Yes','Yes','Yes',60280,1),
(300,'Troy','Sanders','Taft','Manufacturing','Full Time','2007-11-12','No','No','Yes',61150,2),
(301,'Leigh','Newton','Main','Product Development','Full Time','2007-11-26','No','No','Yes',56870,1),
(302,'Alexandra','Vega','North','Quality Assurance','Full Time','2007-12-06','Yes','Yes','Yes',73560,3),
(303,'Cynthia','Harper','North','Account Management','Full Time','2007-12-07','No','No','Yes',34060,2),
(304,'Anthony','Elliott','Taft','Manufacturing','Contract','2007-12-16','No','No','No',80330,4),
(305,'David','Ware','North','Quality Control','Full Time','2007-12-18','No','No','Yes',41380,2),
(306,'Brian','Hess','North','IT','Full Time','2007-12-20','Yes','Yes','No',27560,2),
(307,'Keith','Wolfe','West','Facilities/Engineering','Full Time','2008-01-27','Yes','Yes','Yes',40920,4),
(308,'Timothy','Park','North','Creative','Hourly','2008-02-02','No','No','No',12676,2),
(309,'Dennis','Nguyen','Main','Environmental Compliance','Hourly','2008-02-03','No','No','No',20028,4),
(310,'Sheila','Simon','North','Marketing','Contract','2008-02-11','No','No','No',63850,2),
(311,'Lynn','James','West','Facilities/Engineering','Contract','2008-02-12','No','No','No',83070,3),
(312,'John','Ferguson','Main','Quality Assurance','Half-Time','2008-02-17','No','No','Yes',42740,2),
(313,'Ellen','Fox','North','Account Management','Full Time','2008-02-22','Yes','Yes','No',24340,4),
(314,'Cindy','Sloan','Main','Manufacturing','Full Time','2008-02-24','Yes','Yes','Yes',60100,1),
(315,'Sheri','Castillo','Taft','Account Management','Contract','2008-02-28','No','No','No',36230,2),
(316,'Phillip','Washington','North','Manufacturing','Full Time','2008-03-03','Yes','Yes','No',66430,2),
(317,'Diana','Price','West','Manufacturing','Full Time','2008-03-14','Yes','Yes','No',70020,3),
(318,'Bruce','Torres','South','Quality Assurance','Full Time','2008-03-15','No','No','Yes',32120,1),
(319,'Michael','Burke','Taft','Sales','Contract','2008-03-24','No','No','No',86470,4),
(320,'Leonard','Figueroa','Main','Creative','Contract','2008-04-06','No','No','No',76020,1),
(321,'Liesl','Phillips','South','Training','Full Time','2008-04-11','No','No','Yes',85300,2),
(322,'Jackie','Roberts','West','Manufacturing Admin','Full Time','2008-04-20','Yes','No','No',71400,4),
(323,'Judith','Booker','West','Facilities/Engineering','Contract','2008-04-22','No','No','No',62150,4),
(324,'Elizabeth','Reid','Main','Account Management','Full Time','2008-04-29','No','No','Yes',65880,5),
(325,'William','Sellers','Main','Marketing','Contract','2008-05-15','No','No','No',77930,5),
(326,'Richard','Hart','Watson','Quality Control','Full Time','2008-05-15','Yes','Yes','Yes',86320,4),
(327,'Steven','Todd','Watson','Training','Full Time','2008-05-21','No','Yes','No',37670,3),
(328,'Martin','Ballard','South','IT','Full Time','2008-06-06','Yes','Yes','Yes',87030,3),
(329,'Norman','Schultz','West','Manufacturing','Hourly','2008-06-08','No','No','No',14416,4),
(330,'Jessica','George','North','Quality Control','Contract','2008-06-08','No','No','No',70150,2),
(331,'Daniel','Rodgers','Main','Quality Control','Contract','2008-06-11','No','No','No',63290,5),
(332,'Kathleen','Giles','South','Quality Control','Full Time','2008-06-12','No','No','Yes',46410,2),
(333,'Karen','Pierce','Main','Manufacturing','Contract','2008-06-13','No','No','No',73990,3),
(334,'Meegan','Wheeler','West','Facilities/Engineering','Contract','2008-06-15','No','No','No',83020,4),
(335,'Richard','Brooks','West','Marketing','Half-Time','2008-06-19','No','No','Yes',47705,5),
(336,'Athanasios','Small','Main','Manufacturing','Full Time','2008-06-27','No','No','Yes',45500,3),
(337,'Nancy','Kelley','Main','Quality Control','Contract','2008-06-27','No','No','No',64263,3),
(338,'Preston','McCoy','North','Creative','Half-Time','2008-06-28','Yes','No','No',18655,4),
(339,'Regina','Gutierrez','Main','Account Management','Contract','2008-06-30','No','No','No',33120,2),
(340,'Ewan','Olsen','North','Research/Development','Contract','2008-07-07','No','No','No',60760,2),
(341,'David','Singleton','North','Quality Control','Half-Time','2008-07-09','Yes','Yes','Yes',20075,1),
(342,'Allen','Hancock','North','IT','Full Time','2008-07-11','Yes','Yes','Yes',88850,3),
(343,'Marcia','Mathews','Watson','IT','Full Time','2008-07-14','No','Yes','No',77840,2),
(344,'Lisa','Strong','West','Quality Assurance','Full Time','2008-07-14','Yes','Yes','Yes',54270,3),
(345,'Jeffrey','Charles','Taft','Quality Assurance','Full Time','2008-07-25','No','No','Yes',45110,2),
(346,'Robert','Acosta','Main','Quality Assurance','Full Time','2008-07-28','Yes','Yes','Yes',39000,5),
(347,'Nathan','Robertson','North','Product Development','Half-Time','2008-07-29','No','Yes','No',15005,4),
(348,'Suzanne','Mason','Main','Quality Assurance','Full Time','2008-08-01','No','No','Yes',66824,2),
(349,'Svetlana','Hardy','West','Product Development','Contract','2008-08-04','No','No','No',45040,5),
(350,'Jacklyn','Berry','Taft','Manufacturing','Contract','2008-08-04','No','No','No',64430,4),
(351,'Jonathan','Dawson','West','ADC','Contract','2008-08-07','No','No','No',42540,5),
(352,'Linda','Landry','North','Environmental Health/Safety','Full Time','2008-08-07','No','No','Yes',43680,5),
(353,'Douglas','Gill','North','Marketing','Full Time','2008-08-07','No','Yes','No',45180,5),
(354,'Javier','Tyler','West','Quality Assurance','Full Time','2008-08-13','No','No','Yes',43600,5),
(355,'Tim','Woodward','Main','Product Development','Contract','2008-08-14','No','No','No',26360,4),
(356,'Jeremy','McIntosh','Main','Quality Control','Half-Time','2008-08-15','No','Yes','No',27710,3),
(357,'Jeanne','Hull','West','Marketing','Full Time','2008-08-17','No','No','Yes',47610,4),
(358,'William','Oneal','West','Manufacturing','Contract','2008-08-26','No','No','No',79220,4),
(359,'Danielle','Koch','Watson','Marketing','Contract','2008-08-27','No','No','No',29000,5),
(360,'Diane','Rice','Taft','Manufacturing','Full Time','2008-08-28','Yes','Yes','Yes',24300,3),
(361,'Aria','Newman','South','Quality Control','Full Time','2008-08-31','No','Yes','No',25690,2),
(362,'Polly','Ayala','Main','Account Management','Full Time','2008-09-03','Yes','Yes','Yes',23320,4),
(363,'Douglas','Rivers','West','Account Management','Half-Time','2008-09-05','No','No','Yes',10700,4),
(364,'Yvonne','Randall','North','Marketing','Full Time','2008-09-09','No','Yes','No',86540,4),
(365,'Marilyn','Callahan','South','Manufacturing','Full Time','2008-09-10','No','No','Yes',23330,4),
(366,'Bennet','Schroeder','North','Quality Assurance','Full Time','2008-09-12','Yes','Yes','No',82700,3),
(367,'Boyd','Spencer','Main','Training','Contract','2008-09-18','No','No','No',66580,5),
(368,'Kevin','Merritt','North','Manufacturing','Contract','2008-09-18','No','No','No',63850,2),
(369,'Warren','Medina','North','Human Resources','Full Time','2008-09-26','Yes','Yes','Yes',63190,1),
(370,'Kenneth','Vazquez','North','Product Development','Full Time','2008-09-28','Yes','Yes','Yes',31840,1),
(371,'Jeffrey','Grimes','North','Quality Control','Hourly','2008-10-07','No','No','No',26944,4),
(372,'Kathleen','Flowers','Main','Facilities/Engineering','Full Time','2008-10-14','Yes','Yes','Yes',31910,5),
(373,'Annie','Shepherd','Watson','Account Management','Full Time','2008-10-16','Yes','Yes','Yes',73830,2),
(374,'Jason','Skinner','Main','IT','Full Time','2008-10-23','No','No','Yes',73030,5),
(375,'Melissa','Mills','North','Quality Control','Contract','2008-11-07','No','No','No',75550,3),
(376,'Michael','Stevenson','South','Quality Control','Contract','2008-11-16','No','No','No',78590,1),
(377,'Melyssa','Ellison','North','Quality Control','Half-Time','2008-11-21','Yes','No','No',11230,4),
(378,'David','Fischer','North','Marketing','Full Time','2008-11-26','Yes','No','No',66920,2),
(379,'Rick','Combs','Main','Manufacturing','Contract','2008-11-30','No','No','No',45770,5),
(380,'Lon','Cain','North','Account Management','Full Time','2008-11-30','Yes','No','No',63440,3),
(381,'Catherine','Hampton','Main','Manufacturing','Full Time','2008-12-02','No','No','Yes',81980,2),
(382,'Harold','Summers','North','Manufacturing','Full Time','2008-12-02','Yes','Yes','No',63070,1),
(383,'Sandra','Hubbard','Main','Quality Control','Half-Time','2008-12-05','Yes','Yes','Yes',49545,2),
(384,'Thomas','Blake','South','Quality Control','Contract','2008-12-10','No','No','No',35240,3),
(385,'George','Duncan','North','Quality Assurance','Full Time','2008-12-11','Yes','Yes','Yes',54200,4),
(386,'Brent','Rich','West','Product Development','Contract','2008-12-12','No','No','No',64090,2),
(387,'Jessica','Miller','West','Manufacturing','Half-Time','2008-12-14','Yes','Yes','No',18895,4),
(388,'Michelle','McKee','South','Manufacturing','Hourly','2008-12-16','No','No','No',30416,1),
(389,'Gay','Schneider','Main','Facilities/Engineering','Half-Time','2008-12-17','Yes','Yes','Yes',26890,3),
(390,'Erin','Lawson','Main','Account Management','Full Time','2008-12-20','No','Yes','No',69420,2),
(391,'Charles','Lowery','Main','Marketing','Contract','2008-12-21','No','No','No',74470,3),
(392,'Andrea','Cochran','North','Quality Control','Full Time','2008-12-21','Yes','Yes','Yes',24980,3),
(393,'Richelle','Morris','Main','IT','Full Time','2008-12-22','Yes','Yes','Yes',25830,5),
(394,'Charles','Robles','North','Marketing','Full Time','2008-12-28','No','No','Yes',65250,2),
(395,'Kevin','Bruce','Main','Quality Assurance','Hourly','2008-12-31','No','No','No',26484,5),
(396,'Ted','Wise','North','Product Development','Contract','2009-01-02','No','No','No',40560,5),
(397,'Diane','Gordon','South','Account Management','Contract','2009-01-05','No','No','No',76870,5),
(398,'Laura','Guerrero','Taft','IT','Half-Time','2009-01-06','Yes','No','No',47760,3),
(399,'Holly','Kemp','Watson','Manufacturing','Full Time','2009-01-19','Yes','Yes','No',71030,3),
(400,'Lynne','Wagner','Main','Product Development','Full Time','2009-02-02','Yes','Yes','Yes',72900,3),
(401,'Rena','Bush','South','Marketing','Contract','2009-02-06','No','No','No',81930,5),
(402,'William','Austin','Main','Account Management','Full Time','2009-02-11','Yes','Yes','Yes',62688,2),
(403,'Cynthia','Ortiz','West','Quality Control','Half-Time','2009-02-19','Yes','No','No',23000,4),
(404,'Kyle','Drake','South','Manufacturing','Full Time','2009-02-24','Yes','Yes','Yes',46220,2),
(405,'Max','Mendez','North','Manufacturing','Full Time','2009-03-02','No','No','Yes',67050,4),
(406,'Kevin','Shannon','North','Account Management','Full Time','2009-03-10','Yes','Yes','No',42020,5),
(407,'Lisa','Hunter','West','Account Management','Full Time','2009-03-20','Yes','Yes','Yes',50570,4),
(408,'David','Bell','South','Manufacturing','Contract','2009-03-26','No','No','No',35460,3),
(409,'Laura','Farrell','Watson','Facilities/Engineering','Full Time','2009-03-27','Yes','Yes','Yes',67890,5),
(410,'Kevin','Wade','South','Green Building','Full Time','2009-04-05','Yes','Yes','Yes',71120,4),
(411,'Mark','Nash','Main','Manufacturing','Full Time','2009-04-07','No','Yes','No',71490,5),
(412,'Marcus','Woods','Main','Quality Assurance','Full Time','2009-04-15','No','Yes','No',48490,2),
(413,'Jon','Salinas','North','Quality Control','Full Time','2009-04-16','No','No','Yes',87220,1),
(414,'Christian','Watson','West','Manufacturing','Full Time','2009-04-20','Yes','Yes','No',38940,2),
(415,'Rodney','Luna','North','IT','Full Time','2009-04-21','No','Yes','No',50990,4),
(416,'Roque','Coleman','North','Manufacturing','Full Time','2009-04-24','No','No','Yes',73072,5),
(417,'Benjamin','Walton','North','Environmental Compliance','Full Time','2009-05-01','Yes','Yes','Yes',73440,1),
(418,'John','Bond','North','Environmental Compliance','Half-Time','2009-05-04','No','Yes','No',46095,3),
(419,'Craig','Hodge','Watson','Quality Assurance','Hourly','2009-05-04','No','No','No',23692,4),
(420,'Barbara','Sweeney','North','Quality Control','Full Time','2009-05-22','No','Yes','No',64780,5),
(421,'Anthony','Henderson','South','Quality Control','Full Time','2009-05-28','Yes','Yes','No',68860,2),
(422,'Carlos','Wells','North','Account Management','Full Time','2009-06-02','Yes','Yes','Yes',44650,1),
(423,'Larry','Weber','Watson','ADC','Full Time','2009-06-03','No','Yes','No',72830,2),
(424,'Mary Jo','Johnson','South','Quality Assurance','Full Time','2009-06-04','No','No','Yes',83710,3),
(425,'Stephanie','Cox','Watson','Manufacturing','Full Time','2009-06-11','No','No','Yes',59420,4),
(426,'Juan','Bishop','North','Marketing','Half-Time','2009-06-13','Yes','Yes','Yes',31255,5),
(427,'Blane','Foster','Main','Quality Assurance','Hourly','2009-06-14','No','No','No',36788,4),
(428,'Mark','Conley','South','Manufacturing','Full Time','2009-06-28','Yes','Yes','Yes',41060,3),
(429,'Guy','Vincent','North','Manufacturing','Full Time','2009-06-28','Yes','Yes','Yes',87760,1),
(430,'Jack','Cortez','Main','Green Building','Full Time','2009-07-18','Yes','Yes','Yes',36630,4),
(431,'Chad','Tran','Main','Major Mfg Projects','Half-Time','2009-08-10','No','Yes','No',89780,4),
(432,'Chris','Frazier','Main','Manufacturing','Full Time','2009-08-13','No','No','Yes',24710,2),
(433,'Gary','Watkins','Watson','Account Management','Full Time','2009-08-14','Yes','No','No',61330,2),
(434,'Matt','Stephenson','West','Human Resources','Contract','2009-08-17','No','No','No',71700,2),
(435,'Therese','Pacheco','Main','Green Building','Contract','2009-08-24','No','No','No',35620,4),
(436,'Monica','Hicks','South','Quality Control','Contract','2009-08-29','No','No','No',32880,3),
(437,'Julian','Zimmerman','South','Quality Control','Half-Time','2009-08-30','Yes','No','No',49080,5),
(438,'Kathryn','Blackburn','West','Manufacturing','Contract','2009-09-02','No','No','No',62780,4),
(439,'Justin','McConnell','Main','Manufacturing','Contract','2009-09-03','No','No','No',63310,3),
(440,'Margarita','Roy','Main','Quality Control','Full Time','2009-09-03','No','No','Yes',73850,2),
(441,'Kim','Perez','Main','Manufacturing','Contract','2009-09-09','No','No','No',84170,2),
(442,'Carolyn','Whitehead','North','Facilities/Engineering','Full Time','2009-10-22','Yes','Yes','No',74670,5),
(443,'Ann','Valdez','Taft','Marketing','Contract','2009-10-26','No','No','No',57520,3),
(444,'Pat','Shaw','Main','Account Management','Contract','2009-10-26','No','No','No',56650,1),
(445,'Christine','Golden','Taft','Manufacturing','Full Time','2009-10-31','Yes','Yes','Yes',81010,4),
(446,'Chris','Garrett','West','Quality Control','Full Time','2009-11-05','Yes','Yes','Yes',79380,5),
(447,'Ryan','Ryan','North','Manufacturing','Contract','2009-11-06','No','No','No',40940,2),
(448,'Eric','Floyd','Taft','IT','Contract','2009-11-19','No','No','No',66710,2),
(449,'Robert','Daniel','North','Human Resources','Contract','2009-11-26','No','No','No',60060,2),
(450,'Michael','Collins','West','Manufacturing','Contract','2009-12-06','No','No','No',39680,1),
(451,'Sheryl','Bradshaw','North','Research/Development','Contract','2009-12-12','No','No','No',64720,5),
(452,'Daniel','Malone','North','Environmental Health/Safety','Full Time','2009-12-19','No','No','Yes',69060,1),
(453,'Dennis','Buchanan','North','Quality Control','Contract','2009-12-21','No','No','No',45105,1),
(454,'Michael','Noble','South','Product Development','Full Time','2009-12-27','Yes','No','No',32360,4),
(455,'Michael','Barron','West','Manufacturing','Full Time','2009-12-28','Yes','No','No',34480,3),
(456,'Steven','McClain','West','IT','Full Time','2009-12-30','Yes','No','No',80880,1),
(457,'Robert','Simmons','North','Facilities/Engineering','Half-Time','2010-01-04','Yes','No','No',38920,4),
(458,'Williams','Ward','Main','Account Management','Full Time','2010-01-15','Yes','Yes','Yes',48080,2),
(459,'Rebecca','Murray','Watson','IT','Full Time','2010-01-20','No','No','Yes',80090,2),
(460,'Timothy','Rivera','South','Quality Control','Full Time','2010-01-21','Yes','Yes','Yes',22820,5),
(461,'Brenda','Barnett','North','Human Resources','Full Time','2010-01-25','Yes','Yes','Yes',47350,5),
(462,'David','Dodson','North','Professional Training Group','Half-Time','2010-01-29','No','Yes','No',24815,1),
(463,'Denise','Rodriquez','Main','Professional Training Group','Full Time','2010-01-30','Yes','Yes','Yes',32600,5),
(464,'Francisco','Oliver','Main','Product Development','Full Time','2010-02-03','No','Yes','No',35360,5),
(465,'David','Bradley','Main','Manufacturing','Full Time','2010-02-07','Yes','Yes','Yes',69320,3),
(466,'Richard','Richards','Taft','Manufacturing','Half-Time','2010-02-08','Yes','Yes','No',15260,2),
(467,'Derrell','Espinoza','North','Account Management','Full Time','2010-02-14','No','Yes','No',46110,4),
(468,'Cam','Burton','West','Training','Full Time','2010-02-15','Yes','No','No',58290,5),
(469,'Janet','Gibson','North','Quality Control','Contract','2010-03-02','No','No','No',23340,4),
(470,'Brady','Branch','North','Quality Assurance','Contract','2010-03-03','No','No','No',43320,5),
(471,'Mark','Gray','Taft','Quality Control','Full Time','2010-03-05','Yes','Yes','Yes',44530,2),
(472,'Eric','Stanley','South','Account Management','Half-Time','2010-03-11','Yes','Yes','Yes',45565,1),
(473,'Cynthia','Marsh','North','Quality Control','Full Time','2010-03-11','Yes','Yes','Yes',82370,5),
(474,'Terry','Martin','West','Quality Control','Contract','2010-03-11','No','No','No',86040,5),
(475,'Jennifer','Orr','Taft','IT','Half-Time','2010-03-14','Yes','Yes','Yes',13435,1),
(476,'Marc','Cross','North','Professional Training Group','Hourly','2010-03-16','No','No','No',33232,4),
(477,'Curtis','Watts','West','Professional Training Group','Half-Time','2010-03-18','Yes','Yes','No',39620,5),
(478,'Brian','Walls','West','Account Management','Contract','2010-03-25','No','No','No',23020,4),
(479,'Don','Sims','West','Quality Control','Hourly','2010-03-25','No','No','No',37344,2),
(480,'Lisa','Cooper','North','Quality Assurance','Full Time','2010-03-28','No','No','Yes',29330,5),
(481,'Jonathan','Harrison','Watson','Manufacturing','Hourly','2010-03-30','No','No','No',10572,4),
(482,'Robert','Maldonado','Watson','Professional Training Group','Full Time','2010-04-06','No','No','Yes',43110,2),
(483,'Randy','Nixon','North','Creative','Hourly','2010-04-10','No','No','No',14712,5),
(484,'Karen','Richard','North','Manufacturing','Full Time','2010-04-12','Yes','Yes','Yes',61060,5),
(485,'Mary','Estes','North','Professional Training Group','Full Time','2010-04-13','Yes','Yes','Yes',40940,3),
(486,'Steven','York','North','Facilities/Engineering','Contract','2010-04-17','No','No','No',46670,3),
(487,'Jose','Cummings','Main','Product Development','Half-Time','2010-04-20','Yes','Yes','Yes',39515,5),
(488,'Alan','Craig','Main','Quality Control','Contract','2010-04-20','No','No','No',63610,5),
(489,'Chris','Holloway','North','Account Management','Contract','2010-04-24','No','No','No',85980,2),
(490,'Karen','Guerra','South','IT','Contract','2010-05-05','No','No','No',54000,3),
(491,'Robert','Klein','Main','Quality Control','Full Time','2010-05-06','Yes','Yes','Yes',69510,5),
(492,'Stephanie','Camacho','Watson','Manufacturing','Contract','2010-05-07','No','No','No',80690,3),
(493,'Robert','Holt','North','Quality Assurance','Full Time','2010-05-19','Yes','Yes','Yes',53900,5),
(494,'John','Thompson','Watson','Facilities/Engineering','Half-Time','2010-05-24','Yes','No','No',22535,3),
(495,'Anthony','Gardner','Main','Facilities/Engineering','Contract','2010-05-25','No','No','No',42940,1),
(496,'Stephen','Villarreal','South','Account Management','Full Time','2010-05-29','Yes','Yes','No',88820,2),
(497,'John','Glass','Main','Account Management','Contract','2010-05-31','No','No','No',58650,4),
(498,'Chris','Moreno','West','Quality Assurance','Full Time','2010-06-06','No','No','Yes',72060,2),
(499,'Mark','Conner','West','Manufacturing','Contract','2010-06-13','No','No','No',60040,5),
(500,'Charles','Rojas','West','Manufacturing','Contract','2010-06-21','No','No','No',78520,4),
(501,'Debra','Henson','North','Account Management','Full Time','2010-07-25','Yes','Yes','Yes',64320,5),
(502,'Jody','Lambert','West','Marketing','Half-Time','2010-08-01','No','Yes','No',38575,2),
(503,'Deon','Hale','Main','Product Development','Hourly','2010-08-23','No','No','No',15744,3),
(504,'Melissa','Christian','West','Account Management','Full Time','2010-08-29','Yes','Yes','Yes',24790,3),
(505,'Cheryl','Vance','South','Facilities/Engineering','Contract','2010-09-21','No','No','No',25790,3),
(506,'Mihaela','Kerr','North','Green Building','Full Time','2010-09-22','Yes','Yes','Yes',76440,3),
(507,'Patricia','Morgan','North','Facilities/Engineering','Contract','2010-10-28','No','No','No',79460,5),
(508,'Timothy','Petersen','South','Manufacturing','Full Time','2010-11-10','Yes','Yes','Yes',78170,5),
(509,'Claudia','Wilkerson','Main','Quality Control','Full Time','2010-12-26','No','No','Yes',34990,3),
(510,'Jerry','Townsend','Taft','Product Development','Contract','2011-01-31','No','No','No',56920,4),
(511,'Kevin','Hughes','South','Quality Assurance','Full Time','2011-02-24','No','No','Yes',23190,5),
(512,'Chad','Johns','Watson','Marketing','Full Time','2011-03-01','No','No','Yes',44150,4),
(513,'Rachel','Porter','Main','Facilities/Engineering','Full Time','2011-03-03','Yes','Yes','Yes',41490,5),
(514,'Sperry','Blair','West','Environmental Compliance','Full Time','2011-03-24','Yes','Yes','No',51180,3),
(515,'Johnson','Gallagher','West','Training','Hourly','2011-04-13','No','No','No',10636,4),
(516,'Scott','Jenkins','Main','Account Management','Full Time','2011-04-24','Yes','Yes','Yes',54190,4),
(517,'Gabriel','Chang','Main','Environmental Compliance','Half-Time','2011-05-09','No','No','Yes',28680,1),
(518,'c','c','Main','Account Management','Half-Time','2011-05-23','Yes','No','No',25245,5),
(519,'Brenda','Rhodes','Watson','Manufacturing','Full Time','2011-06-01','No','Yes','No',34690,2),
(520,'Dac Vinh','Calhoun','North','Product Development','Half-Time','2011-06-10','Yes','No','No',21220,3),
(521,'David','Adams','Main','Product Development','Full Time','2011-06-24','Yes','No','No',49260,3),
(522,'Denise','Cunningham','Main','Facilities/Engineering','Full Time','2011-06-26','No','Yes','No',77350,5),
(523,'Keith','McCall','North','Manufacturing','Full Time','2011-06-29','Yes','Yes','Yes',35600,5),
(524,'Jeffrey','West','West','Account Management','Full Time','2011-07-04','Yes','Yes','No',61148,2),
(525,'Lisa','Howard','Main','Marketing','Full Time','2011-07-05','No','No','Yes',45260,4),
(526,'Michelle','Griffith','South','Facilities/Engineering','Contract','2011-07-29','No','No','No',64390,2),
(527,'Cole','Arnold','North','Sales','Full Time','2011-07-31','No','No','Yes',22860,5),
(528,'Ricky','Erickson','North','Quality Control','Contract','2011-07-31','No','No','No',80729,3),
(529,'Robert','Shields','Main','Product Development','Contract','2011-08-01','No','No','No',45830,4),
(530,'Corey','Saunders','South','IT','Full Time','2011-08-11','No','No','Yes',63080,5),
(531,'Keith','Ramirez','North','Quality Control','Full Time','2011-08-15','No','No','Yes',33590,5),
(532,'Jonathan','Stokes','North','Green Building','Contract','2011-08-18','No','No','No',59350,5),
(533,'Todd','Scott','Taft','Account Management','Half-Time','2011-08-19','No','No','Yes',48700,3),
(534,'Anne','Gates','Watson','Product Development','Contract','2011-08-24','No','No','No',45710,3),
(535,'Nicole','Cobb','North','Account Management','Contract','2011-08-24','No','No','No',73190,1),
(536,'ReAnnon','Ray','West','Creative','Contract','2011-08-28','No','No','No',35260,2),
(537,'Eric','Boone','Taft','Major Mfg Projects','Contract','2011-08-28','No','Yes','No',71190,4),
(538,'Tammy','Bowers','South','Sales','Half-Time','2011-08-28','Yes','Yes','Yes',49405,4),
(539,'Heidi','Barker','West','Account Management','Full Time','2011-08-29','Yes','No','No',29760,2),
(540,'Koleen','Smith','Main','Manufacturing','Full Time','2011-09-03','No','No','Yes',86260,3),
(541,'Cris','Eaton','Main','Manufacturing','Full Time','2011-09-04','No','No','Yes',35300,5),
(542,'Marisa','Weiss','West','Facilities/Engineering','Contract','2011-09-07','No','No','No',50550,2),
(543,'Vicki','Swanson','Main','Facilities/Engineering','Full Time','2011-09-08','Yes','Yes','No',38730,1),
(544,'Sally','Neal','North','Account Management','Full Time','2011-09-16','Yes','No','No',72640,3),
(545,'Gerald','Steele','West','Facilities/Engineering','Full Time','2011-09-26','Yes','Yes','Yes',23280,1),
(546,'Dennis','Hanson','Watson','Manufacturing Admin','Contract','2011-09-26','No','No','No',61890,2),
(547,'Phillip','Edwards','Watson','Account Management','Half-Time','2011-09-27','Yes','Yes','Yes',11810,1),
(548,'Joanna','Herring','North','Manufacturing','Contract','2011-10-02','No','No','No',24410,3),
(549,'Patrick','Curtis','North','Product Development','Half-Time','2011-10-03','Yes','No','No',32835,2),
(550,'Donna','Shelton','Main','Manufacturing','Full Time','2011-10-05','No','No','Yes',44270,2),
(551,'Juli','Powell','North','Manufacturing','Half-Time','2011-10-06','Yes','Yes','Yes',46285,5),
(552,'Erik','Huff','Taft','Facilities/Engineering','Full Time','2011-10-14','Yes','No','No',82120,5),
(553,'Grazyna','McBride','North','Manufacturing','Full Time','2011-10-16','Yes','Yes','Yes',73450,3),
(554,'Robin','Hill','Main','Environmental Health/Safety','Hourly','2011-10-17','No','No','No',27484,4),
(555,'John','Savage','Watson','Manufacturing Admin','Full Time','2011-10-17','No','No','Yes',27250,5),
(556,'Jaquelyn','Deleon','North','Facilities/Engineering','Full Time','2011-10-24','No','Yes','No',77580,3),
(557,'Deborah','Bean','Main','Account Management','Full Time','2011-11-05','Yes','Yes','Yes',47340,2),
(558,'Melanie','Spears','West','Marketing','Full Time','2011-11-06','No','Yes','No',70480,4),
(559,'Lynette','Livingston','Taft','Sales','Contract','2011-11-06','No','No','No',74020,2),
(560,'Mark','Cook','Watson','Quality Control','Contract','2011-11-07','No','No','No',47280,1),
(561,'Cleatis','Horton','Main','Quality Assurance','Contract','2011-11-23','No','No','No',21580,3),
(562,'Dan','Everett','Main','Marketing','Half-Time','2011-11-24','No','No','Yes',20040,3),
(563,'Donald','Brown','North','Quality Control','Hourly','2011-12-03','No','No','No',33752,3),
(564,'David','Gonzales','Main','Marketing','Full Time','2011-12-04','No','Yes','No',75780,2),
(565,'Rolin','Evans','North','Product Development','Full Time','2011-12-09','Yes','Yes','Yes',63780,5),
(566,'Matt','Ingram','West','Marketing','Full Time','2011-12-10','Yes','Yes','Yes',48800,4),
(567,'Brian','Stone','North','Facilities/Engineering','Contract','2011-12-11','No','No','No',89310,5),
(568,'Chris','Bennett','Taft','IT','Full Time','2011-12-13','Yes','Yes','Yes',66840,4),
(569,'Timothy','Short','Main','Sales','Contract','2011-12-15','No','No','No',75100,4),
(570,'David','Riley','Taft','Research/Development','Full Time','2011-12-27','Yes','Yes','Yes',46680,1),
(571,'Michael','Harvey','North','Quality Control','Full Time','2012-01-01','Yes','Yes','Yes',58370,5),
(572,'Robert','Anthony','Watson','IT','Half-Time','2012-01-05','Yes','Yes','Yes',16925,1),
(573,'Marc','Myers','South','Quality Control','Contract','2012-01-05','No','No','No',41770,5),
(574,'Carey','Blevins','Taft','Marketing','Full Time','2012-01-07','Yes','Yes','Yes',57560,4),
(575,'Joan','Estrada','Main','Product Development','Full Time','2012-01-11','No','Yes','No',32640,4),
(576,'Nathaniel','Ramsey','West','Account Management','Full Time','2012-01-11','Yes','Yes','No',72700,5),
(577,'Jon','Gregory','West','Human Resources','Full Time','2012-01-12','No','No','Yes',79150,2),
(578,'Catherine','Sawyer','West','IT','Hourly','2012-01-15','No','No','No',15056,5),
(579,'Gustavo','Wiley','Main','Account Management','Contract','2012-01-16','No','No','No',39550,5),
(580,'Jenny','Cannon','West','Account Management','Contract','2012-01-22','No','No','No',57680,4),
(581,'Sandi','Beard','Watson','Quality Control','Half-Time','2012-01-22','No','No','Yes',38105,2),
(582,'Richard','Morrow','North','Account Management','Contract','2012-01-26','No','No','No',60070,2),
(583,'Ralph','Willis','Taft','Quality Control','Full Time','2012-02-01','Yes','Yes','Yes',31690,4),
(584,'Ann','Walters','West','Account Management','Half-Time','2012-02-02','Yes','Yes','No',49355,5),
(585,'Ensley','Brock','Taft','Manufacturing','Full Time','2012-02-05','No','Yes','No',39520,5),
(586,'Dale','Delgado','North','Account Management','Full Time','2012-02-19','No','Yes','No',59150,4),
(587,'Jan','Ramos','Main','Human Resources','Full Time','2012-02-23','Yes','Yes','Yes',66740,2),
(588,'Debbi','Griffin','Main','Marketing','Full Time','2012-02-28','Yes','Yes','Yes',33970,4),
(589,'Shawn','Trujillo','North','Manufacturing','Contract','2012-03-01','No','No','No',88840,5),
(590,'Jenny','Hall','South','Quality Control','Contract','2012-03-03','No','No','No',87830,2),
(591,'Rebecca','Cline','West','IT','Full Time','2012-03-04','No','No','Yes',43410,1),
(592,'Herb','Hines','Main','IT','Hourly','2012-03-04','No','No','No',9180,3),
(593,'Brian','Lyons','Taft','Manufacturing','Half-Time','2012-03-08','Yes','Yes','Yes',46645,5),
(594,'Douglas','Ayers','Main','Manufacturing','Contract','2012-03-14','No','No','No',52940,4),
(595,'Todd','Underwood','Taft','IT','Contract','2012-03-20','No','No','No',39440,4),
(596,'Robert','Peters','Main','Manufacturing','Full Time','2012-03-21','Yes','Yes','No',45480,4),
(597,'Carl','Hatfield','Main','Quality Assurance','Full Time','2012-03-21','Yes','Yes','No',63030,1),
(598,'Marc','Reese','North','Product Development','Full Time','2012-03-22','No','No','Yes',42620,3),
(599,'Chan','Moss','Main','Manufacturing','Contract','2012-03-22','No','No','No',37840,1),
(600,'Troy','Benson','Watson','Manufacturing','Contract','2012-03-25','No','No','No',28260,5),
(601,'Rebecca','McGuire','Watson','Manufacturing','Full Time','2012-03-26','Yes','Yes','Yes',59320,4),
(602,'Michael','Ashley','North','Account Management','Full Time','2012-03-29','No','Yes','No',63206,1),
(603,'Greg','Sanchez','Main','Quality Control','Full Time','2012-03-29','Yes','Yes','Yes',27130,5),
(604,'Evangeline','Hoover','Main','Quality Assurance','Full Time','2012-04-07','No','No','Yes',66440,3),
(605,'Wendy','Patrick','Main','Manufacturing','Contract','2012-04-13','No','No','No',66010,2),
(606,'Doug','Yates','North','IT','Full Time','2012-04-13','Yes','No','No',67230,4),
(607,'Jill','Christensen','Main','Sales','Hourly','2012-04-15','No','No','No',35312,3),
(608,'Matthew','Dorsey','Main','Facilities/Engineering','Full Time','2012-04-22','No','Yes','No',77820,3),
(609,'Bruce','Cohen','Taft','Green Building','Half-Time','2012-04-26','No','No','Yes',46230,2),
(610,'Gretchen','Phelps','South','Environmental Compliance','Contract','2012-04-29','No','No','No',58130,2),
(611,'Jody','Avila','Main','Quality Assurance','Hourly','2012-05-06','No','No','No',33508,4),
(612,'Douglas','Hawkins','Main','Environmental Health/Safety','Half-Time','2012-05-07','No','No','Yes',28625,1),
(613,'Letitia','Velez','North','Quality Assurance','Full Time','2012-05-12','No','No','Yes',34330,3),
(614,'Sherri','Lester','North','Quality Control','Contract','2012-05-14','No','No','No',46570,4),
(615,'Barry','Mack','Main','IT','Full Time','2012-05-16','Yes','Yes','No',77950,4),
(616,'Thomas','Bryan','South','Sales','Full Time','2012-05-24','Yes','No','No',62180,2),
(617,'David','Webster','Watson','Quality Control','Full Time','2012-05-27','No','No','Yes',70730,1),
(618,'Melissa','Flynn','Watson','Research/Development','Hourly','2012-06-03','No','No','No',19044,1),
(619,'Barney','Baker','Taft','Creative','Contract','2012-06-10','No','No','No',80050,2),
(620,'Barry','Barton','North','Professional Training Group','Full Time','2012-06-11','Yes','Yes','Yes',71730,1),
(621,'Gary','Trevino','South','Marketing','Full Time','2012-06-12','Yes','Yes','Yes',62740,4),
(622,'Taslim','King','West','Quality Assurance','Hourly','2012-06-21','No','No','No',30468,2),
(623,'Jason','Keller','North','Quality Control','Contract','2012-06-23','No','No','No',55510,3),
(624,'Terri','Sparks','North','Quality Control','Full Time','2012-06-28','Yes','Yes','Yes',46390,5),
(625,'Andrea','Carey','North','Environmental Health/Safety','Half-Time','2012-07-02','No','No','Yes',10520,4),
(626,'Fiona','Burns','Main','Manufacturing','Hourly','2012-07-04','No','No','No',28424,4),
(627,'Michael','Adkins','North','Quality Assurance','Full Time','2012-07-04','No','No','Yes',24840,1),
(628,'Jean','Warren','Taft','Facilities/Engineering','Full Time','2012-07-05','Yes','No','No',74710,2),
(629,'Jeff','Bridges','West','Manufacturing','Full Time','2012-07-08','Yes','Yes','Yes',43820,2),
(630,'Cristin','Harrell','North','Quality Assurance','Full Time','2012-07-11','No','Yes','No',80260,3),
(631,'Dana','Lang','Main','Quality Control','Full Time','2012-07-14','Yes','Yes','Yes',24200,5),
(632,'John','Gentry','North','Professional Training Group','Full Time','2012-07-15','Yes','Yes','Yes',87950,4),
(633,'Steven','Alvarez','Taft','Quality Assurance','Contract','2012-07-17','No','No','No',89450,2),
(634,'Scott','Williams','Taft','Professional Training Group','Contract','2012-07-21','No','No','No',49070,3),
(635,'Hector','Taylor','West','ADC','Half-Time','2012-07-25','Yes','Yes','No',26795,4),
(636,'Debra','McDonald','North','Facilities/Engineering','Full Time','2012-07-26','No','Yes','No',68910,5),
(637,'Debra','Gibbs','North','Manufacturing','Full Time','2012-08-02','No','Yes','No',44260,1),
(638,'Jingwen','Leach','North','Marketing','Hourly','2012-08-09','No','No','No',36844,4),
(639,'Brian','Greer','Taft','IT','Contract','2012-08-19','No','No','No',84300,1),
(640,'Tito','Holmes','Main','Marketing','Half-Time','2012-08-23','No','Yes','No',13090,4),
(641,'Kevin','Burnett','Main','Account Management','Full Time','2012-08-23','Yes','No','No',86500,1),
(642,'John','Lamb','Main','Professional Training Group','Full Time','2012-08-24','Yes','No','No',35320,3),
(643,'Traci','Brady','Main','Manufacturing','Full Time','2012-09-02','Yes','Yes','Yes',47440,3),
(644,'Norman','Hunt','Main','Quality Control','Full Time','2012-09-05','Yes','Yes','Yes',86640,3),
(645,'Ronald','Casey','Taft','Quality Control','Contract','2012-09-06','No','No','No',42990,4),
(646,'Tamara','Norris','West','Facilities/Engineering','Full Time','2012-09-21','Yes','No','No',49810,2),
(647,'Phillip','Parsons','North','Professional Training Group','Half-Time','2012-09-22','Yes','No','No',16015,3),
(648,'Scott','Lynch','Main','Account Management','Full Time','2012-10-04','Yes','Yes','Yes',24090,4),
(649,'Sumed','Williamson','Main','Manufacturing','Contract','2012-10-18','No','No','No',57110,3),
(650,'Julie','Morrison','Main','Quality Control','Full Time','2012-10-18','Yes','Yes','Yes',23030,4),
(651,'Joseph','Schwartz','Main','Quality Control','Full Time','2012-10-18','Yes','No','No',40260,5),
(652,'Randy','Romero','Main','Major Mfg Projects','Full Time','2012-10-28','Yes','Yes','Yes',89140,1),
(653,'Carol','Dalton','West','Research Center','Contract','2012-10-30','No','No','No',85510,4),
(654,'Andrew','Hansen','North','Quality Control','Half-Time','2012-11-03','No','No','Yes',13455,2),
(655,'Linda','Morales','West','Quality Control','Contract','2012-11-13','No','No','No',34680,5),
(656,'Teri','Roman','Main','Account Management','Contract','2012-11-14','No','No','No',79380,1),
(657,'Davin','Gross','Main','Product Development','Full Time','2012-11-17','No','No','Yes',32140,2),
(658,'Theo','Hamilton','Taft','Marketing','Full Time','2012-11-19','Yes','Yes','Yes',22900,1),
(659,'Jeffrey','Ortega','South','Quality Control','Contract','2012-11-25','No','No','No',26020,5),
(660,'Robert','Patterson','Main','Research Center','Contract','2012-11-26','No','No','No',66132,4),
(661,'Brandyn','Lane','Main','Quality Assurance','Contract','2012-12-03','No','No','No',46650,2),
(662,'Mark','Jordan','North','Product Development','Contract','2012-12-06','No','No','No',22320,2),
(663,'Dean','Contreras','Main','Professional Training Group','Full Time','2012-12-22','Yes','Yes','Yes',69400,5),
(664,'Donald','Perkins','West','Creative','Full Time','2012-12-29','No','No','Yes',37620,5),
(665,'Jeff','Murphy','Taft','Quality Assurance','Full Time','2013-01-05','Yes','Yes','Yes',67920,4),
(666,'Lisa','Hodges','North','IT','Full Time','2013-01-08','No','Yes','No',61470,5),
(667,'Jim','Webb','Main','Professional Training Group','Full Time','2013-01-11','No','No','Yes',77720,3),
(668,'William','William','Main','Account Management','Full Time','2013-01-11','No','Yes','No',77740,1),
(669,'Kuyler','Goodman','North','Facilities/Engineering','Half-Time','2013-01-23','Yes','Yes','No',13800,3),
(670,'Khurrum','Brewer','Main','Training','Half-Time','2013-01-25','Yes','Yes','No',30445,1),
(671,'Donald','Holland','South','Training','Hourly','2013-02-02','Yes','Yes','Yes',29070,3),
(672,'David','Graham','North','Quality Control','Contract','2013-02-15','No','No','No',62480,5),
(673,'Sonia','Alvarado','Main','IT','Half-Time','2013-02-22','Yes','Yes','No',35045,4),
(674,'Stephen','Lopez','Main','Quality Control','Contract','2013-02-26','No','No','No',61134,4),
(675,'William','Frank','Main','Account Management','Full Time','2013-03-02','Yes','Yes','No',54500,5),
(676,'Julia','Bartlett','Watson','Professional Training Group','Full Time','2013-03-05','No','Yes','No',44560,2),
(677,'Lowell','Mann','Main','Account Management','Contract','2013-03-07','No','No','No',52750,1),
(678,'Shawn','Herrera','Main','Account Management','Full Time','2013-03-18','No','Yes','No',79400,4),
(679,'Angela','Mullins','Main','Facilities/Engineering','Full Time','2013-03-19','No','No','Yes',85920,4),
(680,'Anthony','Garza','Main','Professional Training Group','Full Time','2013-03-28','Yes','Yes','Yes',81530,5),
(681,'Justin','Monroe','Watson','Quality Assurance','Full Time','2013-04-09','No','No','Yes',63050,3),
(682,'Kelly','Wyatt','West','Environmental Compliance','Full Time','2013-04-12','Yes','Yes','No',41350,2),
(683,'Lisa','Hopkins','North','Quality Control','Contract','2013-04-23','No','No','No',57500,1),
(684,'Charles','Woodard','North','Account Management','Full Time','2013-05-04','Yes','No','No',71680,4),
(685,'Robert','Hammond','Main','Facilities/Engineering','Full Time','2013-05-06','Yes','Yes','No',61400,5),
(686,'Dennis','Wong','Main','IT','Full Time','2013-05-09','Yes','Yes','Yes',43580,5),
(687,'Matt','Ford','West','Account Management','Contract','2013-05-09','No','No','No',50840,4),
(688,'Michael','Hartman','West','Professional Training Group','Full Time','2013-05-19','No','No','Yes',44620,5),
(689,'Maria','Fisher','North','Quality Assurance','Full Time','2013-06-04','Yes','Yes','Yes',54830,1),
(690,'Lori','Knox','North','Marketing','Full Time','2013-06-06','Yes','Yes','No',87120,3),
(691,'Timothy','Allison','Main','Creative','Full Time','2013-06-13','No','Yes','No',82500,5),
(692,'Fredrick','Rios','North','Manufacturing','Full Time','2013-06-13','Yes','No','Yes',56900,5),
(693,'Susan','Maynard','South','Sales','Full Time','2013-06-17','Yes','Yes','Yes',39110,5),
(694,'Greg','Bullock','North','Facilities/Engineering','Hourly','2013-06-20','No','No','No',14568,3),
(695,'Brenda','Ellis','West','IT','Full Time','2013-06-20','No','No','Yes',43190,2),
(696,'Christopher','Castro','Main','Manufacturing','Full Time','2013-07-01','Yes','Yes','Yes',52940,4),
(697,'Greg','Reeves','West','IT','Full Time','2013-07-06','Yes','Yes','Yes',26360,1),
(698,'Adam','Frost','West','Account Management','Contract','2013-07-08','No','No','No',47590,3),
(699,'Matthew','Sutton','North','IT','Full Time','2013-07-12','Yes','Yes','Yes',79770,4),
(700,'Henrietta','Herman','Main','Manufacturing','Full Time','2013-07-18','Yes','No','No',60380,4),
(701,'George','Fitzgerald','West','Account Management','Contract','2013-07-28','No','No','No',60550,2),
(702,'Angela','Higgins','West','Account Management','Half-Time','2013-08-10','Yes','Yes','Yes',46380,3),
(703,'Dana','Pitts','Taft','Facilities/Engineering','Contract','2013-08-17','No','No','No',64460,1),
(704,'Paul','Leonard','North','Quality Control','Full Time','2013-08-20','Yes','Yes','No',46550,4),
(705,'Carrie','Dyer','North','Quality Assurance','Full Time','2013-08-24','Yes','Yes','Yes',65571,3),
(706,'Daniel','Hogan','North','Marketing','Full Time','2013-09-03','Yes','Yes','No',60560,4),
(707,'Robert','Prince','West','Marketing','Full Time','2013-09-10','Yes','Yes','Yes',37020,2),
(708,'Mark','Lara','West','Product Development','Half-Time','2013-09-17','Yes','Yes','Yes',34110,4),
(709,'Dwight','Owens','North','Quality Assurance','Full Time','2013-09-19','Yes','Yes','Yes',68470,4),
(710,'Craig','Beck','North','Environmental Compliance','Full Time','2013-09-21','Yes','Yes','Yes',46220,3),
(711,'Alexander','Greene','West','Manufacturing','Full Time','2013-09-28','No','No','Yes',58910,1),
(712,'Brenda','Fuller','Main','Manufacturing','Full Time','2013-09-29','No','No','Yes',26190,5),
(713,'Scott','McCullough','North','Sales','Full Time','2013-10-19','Yes','Yes','Yes',48550,5),
(714,'Debbie','Richardson','North','Facilities/Engineering','Full Time','2013-10-24','Yes','Yes','No',31830,3),
(715,'Christopher','Joseph','South','Manufacturing','Hourly','2013-10-29','No','No','No',22344,4),
(716,'Patricia','Buckel','Taft','Marketing','Full Time','2013-11-12','Yes','Yes','No',73930,1),
(717,'David','English','North','Manufacturing','Contract','2013-11-21','No','No','No',32190,3),
(718,'John','Hickman','West','Facilities/Engineering','Full Time','2013-12-03','Yes','Yes','Yes',71150,2),
(719,'Kes','Bowen','West','Quality Control','Contract','2013-12-06','No','No','No',59128,4),
(720,'Tonya','Davis','Main','Quality Control','Full Time','2013-12-23','Yes','Yes','No',62780,3),
(721,'Matthew','Walsh','North','IT','Contract','2013-12-28','No','No','No',32650,1),
(722,'Shannon','Mitchell','Main','Manufacturing','Contract','2014-01-05','No','No','No',49530,2),
(723,'Duane','Snyder','West','Quality Control','Contract','2014-01-06','No','No','No',70300,3),
(724,'John','Lloyd','North','Creative','Full Time','2014-01-09','No','No','Yes',82760,4),
(725,'Jesse','Wilkins','North','Training','Full Time','2014-01-17','Yes','Yes','Yes',79760,5),
(726,'Susan','Carr','South','Manufacturing Admin','Full Time','2014-01-18','Yes','Yes','Yes',39160,3),
(727,'Brian','Duran','Taft','ADC','Hourly','2014-02-01','No','No','No',35680,2),
(728,'Michael','Walter','Main','Quality Assurance','Full Time','2014-02-07','Yes','No','No',86240,1),
(729,'Dennis','Clarke','South','Product Development','Full Time','2014-02-27','Yes','Yes','Yes',64510,3),
(730,'Melanie','Olson','North','IT','Full Time','2014-03-05','Yes','No','No',75370,2),
(731,'Shaun','Peterson','Main','IT','Full Time','2014-03-08','Yes','No','No',46910,3),
(732,'Aron','Harrington','Watson','Quality Assurance','Half-Time','2014-03-17','No','No','Yes',25885,5),
(733,'Tracy','Poole','North','Facilities/Engineering','Full Time','2014-03-22','No','No','Yes',71670,4),
(734,'Kirk','Ball','North','Sales','Full Time','2014-03-31','Yes','Yes','No',87980,1),
(735,'Madelyn','Obrien','North','Quality Assurance','Contract','2014-04-10','No','No','No',55690,2),
(736,'Mark','Rose','Watson','Quality Control','Full Time','2014-04-17','No','Yes','No',32160,3),
(737,'Michelle','Lowe','Taft','Product Development','Full Time','2014-04-19','No','No','Yes',46340,5),
(738,'Karin','Terry','North','IT','Full Time','2014-04-24','Yes','Yes','No',68010,1),
(739,'Christopher','Padilla','Main','Product Development','Contract','2014-05-15','No','No','No',81070,5),
(740,'Daniel','Solis','Main','Account Management','Contract','2014-05-15','No','No','No',44720,2),
(741,'Karen','Garcia','Main','Quality Control','Full Time','2014-05-23','Yes','Yes','No',59490,3);
