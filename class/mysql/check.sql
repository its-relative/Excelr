use batch25;
select dep_id, count(dep_id) from myemp group by dep_id order by dep_id;
select dep_id, min(salary), max(salary) from myemp group by dep_id order by dep_id;
#select * from movies inner join members on id =  memid;

