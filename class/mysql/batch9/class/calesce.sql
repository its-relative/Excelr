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
select t1.pid, t1.name, t2.price from table1 as t1
outer join table2 as t2
on t1.pid = t2.pid;
