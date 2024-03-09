use batch9;

select *,
	case when `salary in k` >=90 then 'high'
		when `salary in k` >= 70 then 'medium'
		else "low"
    end as Type1
			from practicecase;
            
 
            
select *,
	case when `salary in k` >=90 then `salary in k`*1.10 
		when `salary in k` >=70 then `salary in k`*1.08
        else `salary in k`*1.05
	end as NewSalaries
			from practicecase;
            
            
use practicing;

	create table dummy(age int check (age > 18));
	insert into dummy values(32);
	insert into dummy values(17);

create table dummy2 (id int not null, `name` varchar(30));
insert into dummy2 values(1,"Sap");

create table dummy3 (id int, name varchar(15) not null);
desc dummy3;

-- defualt 
create table coaching(roll_no int, name varchar(20) not null, course varchar(20) default ("DA"));
insert into coaching(roll_no, name) values(501,"Dinesh"); -- won't work without specifying column names or else column count doesn't match 
insert into coaching values(502,"Ravi","Sql");
select * from coaching;

-- practice
create table coach(sid int unique, sname varchar(20) not null, sage int check(sage >= 20),course varchar(20) default ("DA"),fees int); 
insert into coach values(1,"Ram",23,"python",1500);
insert into coach values(1,"Ram",13,"python",2500);

-- joins
drop table if exists table1;
drop table if exists table2;

create table table1(pid char(2),name varchar(20));
create table table2(pid char(2), price int);

insert into table1(pid,name) values("a","tv"),("b","phone"),("c","ac"),("d","refrigrator"),("e","LED"),("f","Microwave");
insert into table2(pid,price) values("a",500),("b",600),("c",700),("d",900),("i",1100),("x",1200);

select * from table1;
select * from table2;

-- inner join

select t1.pid, t1.name, t2.price from table1 as t1
inner join table2 as t2
on t1.pid = t2.pid;

-- without alias 
-- select table1.pid, table1.name, table2.price from table1 inner join table2 on table1.pid = table2.pid;

-- left join
select t1.pid, t1.name, t2.price from table1 as t1
left join table2 as t2
on t1.pid = t2.pid;

-- right join
select t1.pid, t1.name, t2.price from table1 as t1
right join table2 as t2
on t1.pid = t2.pid;

-- full outer join
-- select t1.pid, t1.name, t2.price from table1 as t1
-- outer join table2 as t2
-- on t1.pid = t2.pid;

-- self join

use batch9;
-- show tables;
select * from myemp;
select e.emp_id,concat(e.first_name," ",e.last_name) full_name,m.mgr_id, concat(m.first_name," ",m.last_name) mgr_name from myemp e left join myemp m on e.mgr_id = m.emp_id;

-- employees without any people assigned
select e.emp_id,concat(e.first_name," ",e.last_name) full_name,m.mgr_id, concat(m.first_name," ",m.last_name) mgr_name from myemp e right join myemp m on e.mgr_id = m.emp_id;

use batch9;
show tables;
select * from authors;
select * from books;

select b.title, a.name from books b left join authors a on b.authorid = a.authorid;

-- savepoint, commit and rollback

set autocommit = 0;

-- Start a transaction
START TRANSACTION;



create table your_table (column1 varchar(8), column2 varchar(8));
-- Execute some queries
INSERT INTO your_table (column1, column2) VALUES ('value1', 'value2');

-- Set a savepoint
SAVEPOINT my_savepoint;

insert into your_table values('value3',"value4");

-- Execute more queries
UPDATE your_table SET column1 = 'new_value' WHERE column2 = 'value2';

-- Roll back to the savepoint
ROLLBACK TO my_savepoint;

-- Commit or Rollback the transaction
COMMIT; -- or ROLLBACK; if you want to discard all changes

select * from your_table;


-- TCL

select * from myemp;

select avg(salary) from myemp;

select * from myemp where salary > (select avg(salary) from myemp) order by salary desc limit 2;
select * from myemp order by salary asc limit 2;

select * from myemp limit 2,1;
select * from myemp where salary < (select max(salary) from myemp) limit 1;


-- select a.name, count(b.title) as coun from books b left join authors a where coun > (select count(b.title) as cnt from books b where cnt > 4 left join authors a on b.authorid = a.authorid group by name);
select * from books;
select a.name , count(b.title) as cnt from books b left join authors a on b.authorid = a.authorid group by name having cnt > 3;

select a.name , b.title, b.authorid from books b left join authors a on b.authorid = a.authorid where a.name = "Chetan Bhagat" or a.name = "Oscar Wilde";


select * from authors;
select * from books where authorid = (select authorid from authors where name = "Chetan bhagat");
select * from books where authorid in (select authorid from authors where name = "Chetan Bhagat" or name = "Oscar Wilde" group by authorid) ;
select authorid from authors where name = "Chetan Bhagat" or name = "Oscar Wilde";

-- delimiters
use batch9;
show tables;

select * from 1data;

select * from 2data;

delimiter ;

show tables ;

-- stored procedures

use batch9;

drop procedure get_emps;

delimiter //



create procedure get_emps()
begin
	select * from myemp;
end //

delimiter ;

show procedure status;

call get_emp();

show tables;
select * from myemp;


-- stored procedure
drop procedure getsal;
delimiter //

create procedure getsal(in sal int)
begin
	select * from myemp where salary = sal;
end//
delimiter ;

call getsal(9000);

-- multiple inputs
drop procedure if exists getName;
delimiter //
create procedure getName(in sal int, in firstname varchar(20))
begin
select * from myemp where salary = sal and first_name = firstname;
end//
delimiter ;

call getName(9000,"Daniel");

-- functions 

use batch9;
select * from myemp;

-- create view
create view show_data as
	select * from myemp where dep_ID = 60;
    
select * from show_data;

-- sequence object
-- create table sid,sname,scourse
-- datatype int, varchar, varchar
drop table if exists object;

create table object (sid int Primary key auto_increment, sname varchar(8), scourse Text);
insert into object(sid, sname,scourse) values(0, "Sapeksh","Mysql");
insert into object(sname,scourse) values("Ram","PowerBi");
select * from object;

alter table object auto_increment = 100; 

-- delete and truncate

set sql_safe_updates = 0;
truncate table object;
delete from object;